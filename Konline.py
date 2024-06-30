import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec

# device setup
from metric import cal_pro
import logging
import Klink

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('Konline')
    parser.add_argument('--data_path', type=str, default='../datasets/MVTec-AD')
    parser.add_argument('--save_path', type=str, default='./btad_result')
    parser.add_argument('--random_seed', type=int, default=1025)
    parser.add_argument("--research", type=bool, default=True)
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument("--model_path", type=str,
                        default="")
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = './ng_online' + str(args.random_seed) + '.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        # model = torch.load(args.model_path)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))
    setup_seed(args.random_seed)

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]
    avi = []
    avp = []
    avpr = []

    for class_name in mvtec.CLASS_NAMES:
        total_roc_auc = []
        total_pixel_roc_auc = []
        total_pro_auc = []

        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=10, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=10, pin_memory=True, shuffle=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        # test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath) or args.research:
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
                break
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            # B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.numpy()
            embedding_vectors = embedding_vectors.transpose(0, 2, 3, 1)
            B, H, W, C = embedding_vectors.shape
            embedding_vectors = embedding_vectors.reshape(B * H * W, C)

            kcluster = Klink.Klinks(embedding_vectors)
            if kcluster.patch_core is True:
                kcluster.fit_network(epochs=20)
            else:
                kcluster.fit_network()
            kcluster.cluster_label()
            # with open(train_feature_filepath, 'wb') as f:
            #     pickle.dump(kcluster, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                kcluster = pickle.load(f)

        # extract test set features
        gt_list_total = []
        gt_mask_list_total = []
        image_score_list = []
        score_map_list = []
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name, position=1):
            # 清空每次online数据
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            gt_list = []
            gt_mask_list = []
            test_imgs = []

            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            gt_list_total.extend(y.cpu().detach().numpy())
            gt_mask_list_total.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = test_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

            # calculate distance matrix
            # B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.permute(0, 2, 3, 1)
            B, H, W, C = embedding_vectors.size()
            embedding_vectors = embedding_vectors.contiguous().view(B * H * W, C).numpy()
            dist_list = []
            ranking = kcluster.gpu_cal_dist(embedding_vectors[:, :])
            _, topk_indexes = torch.topk(ranking, k=1, dim=1, largest=False, sorted=True)

            step = 500
            for j in tqdm(range((B * H * W) // step + 1), '| anomaly detection | test | %s |' % class_name, position=0):
                mean_list = []
                conv_inv_list = []
                if j == (B * H * W) // step:
                    sum_sample = (B * H * W) % step
                else:
                    sum_sample = step
                for i in range(j * step, j * step + sum_sample):
                    # 取获胜节点邻域内最小的马氏距离
                    s1 = kcluster.node_list[topk_indexes[i][0]]
                    mean = kcluster.network.nodes[s1]['vector'].reshape(1, -1)
                    cov_inv = kcluster.network.nodes[s1]["cov_inv"].reshape(1, C, C)
                    mean_list.append(mean)
                    conv_inv_list.append(cov_inv)
                mean_matirx = np.concatenate(mean_list)
                conv_inv_matrix = np.concatenate(conv_inv_list)
                dist = Matrix_Mahalanobis(embedding_vectors[j * step:j * step + sum_sample], mean_matirx,
                                          conv_inv_matrix)
                dist_list.append(dist)

            dist_list = np.concatenate(dist_list).reshape(B, H, W)

            # upsample
            dist_list = torch.tensor(dist_list)
            score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                      align_corners=False).squeeze().numpy()

            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            # Normalization
            # score_map_list.append(score_map.reshape(-1, 224, 224))
            max_score = score_map.max()
            min_score = score_map.min()
            scores = (score_map - min_score) / (max_score - min_score)

            # calculate image-level ROC AUC score
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            gt_mask = np.asarray(gt_mask_list)

            if (1 in gt_list) and (1 in gt_mask.flatten()) and (0 in gt_list):
                gt_list = np.asarray(gt_list)
                # print(gt_list, img_scores)
                fpr, tpr, _ = roc_curve(gt_list, img_scores)
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                total_roc_auc.append(img_roc_auc)
                logger.info('image ROCAUC: %.3f' % (img_roc_auc))
                # fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

                # get optimal threshold

                precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
                a = 2 * precision * recall
                b = precision + recall
                f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
                threshold = thresholds[np.argmax(f1)]

                # calculate per-pixel level ROCAUC
                fpr, tpr, roc_thresholds = roc_curve(gt_mask.flatten(), scores.flatten())
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                total_pixel_roc_auc.append(per_pixel_rocauc)
                logger.info('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

                # cal pro
                pro_score = cal_pro(gt_mask_list, scores, roc_thresholds, fpr)
                total_pro_auc.append(pro_score)
                logger.info('%s pro_score : %.3f' % (class_name, pro_score))

                # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
                # os.makedirs(save_dir, exist_ok=True)
                # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

            kcluster.online_fit(embedding_vectors)
            # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        #
        # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        # os.makedirs(save_dir, exist_ok=True)
        # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        ####logger
        logger.info('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
        # logger.info(total_roc_auc)
        # fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
        # fig_img_rocauc.legend(loc="lower right")

        logger.info('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
        # fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
        # logger.info(total_pixel_roc_auc)
        # fig_pixel_rocauc.legend(loc="lower right")

        logger.info('Average PRO: %.3f' % np.mean(total_pro_auc))
        # logger.info(total_pro_auc)
        avi.append(np.mean(total_roc_auc))
        avp.append(np.mean(total_pixel_roc_auc))
        avpr.append(np.mean(total_pro_auc))

        logger.info("\n\n")
    print(np.mean(avi), np.mean(avp), np.mean(avpr))

    #
    # fig.tight_layout()
    # fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)



def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def Matrix_Mahalanobis(embedding, mean, cov):
    B, C = embedding.shape
    delta = (embedding - mean).reshape(B, 1, C)
    temp = np.einsum("ijk,ikm->ijk", delta, cov)
    dist = np.einsum("ijk,ikn->ijn", temp, delta.transpose(0, 2, 1)).squeeze()
    dist[dist < 0] = 0
    return np.sqrt(dist)


def Matrix_Mahalanobis_GPU(embedding, mean, cov):
    B, C = embedding.shape
    delta = (embedding - mean).reshape(B, 1, C)
    delta = torch.from_numpy(delta).cuda()
    cov = torch.from_numpy(cov).cuda()
    temp = torch.einsum("ijk,ikm->ijk", delta, cov)
    dist = torch.einsum("ijk,ikn->ijn", temp, delta.permute(0, 2, 1)).squeeze()
    dist = dist.numpy()
    dist[dist < 0] = 0
    return np.sqrt(dist)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, gt, color=(1, 0, 0), outline_color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))
        fig_img.subplots_adjust(top=0.95, bottom=0.05, right=0.9, wspace=0.05)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        truth = (gt * 255).astype(np.uint8)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        # ax_img[0].title.set_text('Image')
        # ax_img[2].imshow(gt, cmap='gray')
        # ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        # ax_img[2].title.set_text('Predicted heat map')
        # ax_img[3].imshow(mask, cmap='gray')
        # ax_img[3].title.set_text('Predicted mask')
        alpha_mask = np.zeros([224, 224]).flatten()
        alpha_mask[np.where(truth.flatten() == 255)] = 0.4
        alpha_mask = alpha_mask.reshape((224, 224))
        ax_img[1].imshow(vis_img)
        ax_img[1].imshow(truth, cmap='jet', alpha=alpha_mask)
        # ax_img[1].imshow(truth, cmap='jet', alpha=0.5, interpolation='none')

        # ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.1
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}.pdf'.format(i)), dpi=100, bbox_inches='tight')
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}.png'.format(i)), dpi=600, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
