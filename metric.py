import numpy as np
from skimage import measure
import os
import matplotlib.pyplot as plt


def cal_pro(gd_mask_list, pre_list, roc_thresholds, fpr):
    fpr_list = np.linspace(0.0, 0.3, num=100)
    pro_list = []
    for fpr_i in fpr_list:
        ind = np.argmin(fpr < fpr_i)
        th_fpr = roc_thresholds[ind]
        pre_mask_list = []
        overlap_list = []
        for pre in pre_list:
            pre_mask_list.append(pre >= th_fpr)
        for gd_mask, pre_mask in zip(gd_mask_list, pre_mask_list):
            gd_mask = gd_mask.squeeze()
            assert gd_mask.shape == pre_mask.shape
            gd_region = measure.label(gd_mask, connectivity=2)
            uni_label = np.unique(gd_region)

            for l in uni_label:
                if l != 0:
                    gd_region_mask = (gd_region == l).astype(np.int32)
                    overlap = 1.0 * np.sum(gd_region_mask * pre_mask) / np.sum(gd_region_mask)
                    overlap_list.append(overlap)
        pro_list.append(np.mean(overlap_list))
    pro_score = np.mean(pro_list)
    return pro_score
