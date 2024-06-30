# coding: utf-8
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm



class Klinks:
    def __init__(self, input_data, init_node=56 * 56, age_max=25, mode=False):
        self.init_node = init_node
        self.age_max = age_max
        self.data = input_data
        self.node_list = []
        self.online_data = None
        self.network = nx.Graph()
        self.centeroid = None
        self.data_len = self.data.shape[0]
        self.units_created = 0
        self.sparse_thresh = 0
        self.patch_core = mode
        if self.patch_core is True:
            pass
        else:
            rand_init = np.random.choice(self.data_len, size=init_node)
        for i in rand_init:
            w_i = self.data[i]
            self.network.add_node(self.units_created, vector=w_i, error=0, threshold=float("-inf"),
                                  cov=0, nodes_num=0, cov_inv=0)
            self.units_created += 1
        plt.style.use('ggplot')

    def gpu_cal_dist(self, data):
        observation = torch.from_numpy(data)
        B, L = observation.shape
        centeroid = torch.from_numpy(self.centeroid)
        centeroid = centeroid.t().unsqueeze(0).cuda()
        dist_list = []
        batch_size = 500
        for d_idx in range(B // batch_size + 1):
            if d_idx == B // batch_size:
                test_batch = observation[d_idx * batch_size:]
                test_batch = test_batch.cuda()
            else:
                test_batch = observation[d_idx * batch_size:d_idx * batch_size + batch_size]
                test_batch = test_batch.cuda()
            dist_matrix = torch.pairwise_distance(test_batch.unsqueeze(-1),
                                                  centeroid)
            dist_list.append(torch.sqrt(dist_matrix).cpu())
        dist_matrix = torch.cat(dist_list)
        return dist_matrix

    def fit_network(self, epochs=10):
        for epoch in tqdm(range(epochs), "| epochs |", position=1):
            center_list = []
            node_idx = []

            for u in self.network.nodes():
                center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
                node_idx.append(u)
            self.node_list = np.asarray(node_idx)
            self.centeroid = np.concatenate(center_list)
            dist_matrix = self.gpu_cal_dist(self.data)
            topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
            topk_indexes = topk_indexes.numpy()
            topk_values = topk_values.numpy()

            for u in self.network.nodes():
                self.network.nodes[u]['nodes_num'] = 0
                self.network.nodes[u]['error'] = 0
                self.network.nodes[u]['vector'] = 0

            for i in tqdm(range(len(self.data)), "| processing data |", position=0):
                s_1 = node_idx[topk_indexes[i][0]]
                s_2 = node_idx[topk_indexes[i][1]]

                self.network.add_edge(s_1, s_2, age=0)
                self.network.nodes[s_1]['vector'] += self.data[i]
                self.network.nodes[s_1]['nodes_num'] += 1
                self.network.nodes[s_1]['error'] += topk_values[i][0]

                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age'] + 1)
            nodes_to_remove = []
            node_num = 0
            for u in self.network.nodes():
                if type(self.network.nodes[u]['vector']) is np.ndarray:
                    self.network.nodes[u]['vector'] = self.network.nodes[u]['vector'] / (self.network.nodes[u][
                                                                                             'nodes_num'] + 1e-3)
                    node_num += 1
                else:
                    nodes_to_remove.append(u)
            self.del_nul_nodes(nodes_to_remove)
            self.prune_connections()
            print(node_num)


    def online_fit(self, online_data):
        self.online_data = online_data

        self.update_threshold()
        topk_values, topk_indexes = self.online_gpu_cal_dist(self.online_data)
        topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()


        label_matrix = [[] for i in range(self.units_created)]
        new_node_matrix = [0 for i in range(self.units_created)]

        C = self.online_data.shape[1]
        I = np.identity(C)

        temp_list = []
        temp_dist_list = []
        count = 0
        cout1 = 0
        for i in tqdm(range(len(self.online_data)), "| processing data |"):
            s_1 = self.node_list[topk_indexes[i][0]]
            s_2 = self.node_list[topk_indexes[i][1]]

            if topk_values[i][0] > self.network.nodes[s_1]['threshold']:
                if topk_values[i][1] > self.network.nodes[s_2]['threshold']:
                    count += 1
                    temp_list.append(self.online_data[i:i + 1])
                    temp_dist_list.append(topk_values[i][0])
                else:
                    cout1 += 1

                    temp_list.append(self.online_data[i:i + 1])
                    temp_dist_list.append(topk_values[i][0])
                continue
            self.network.add_edge(s_1, s_2, age=0)
            label_matrix[s_1].append(i)
            new_node_matrix[s_1] += 1

            self.network.nodes[s_1]['error'] += topk_values[i][0]
            for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                self.network.add_edge(u, v, age=attributes['age'] + 1)
        nodes_to_remove = []
        node_num = 0
        vector_num = 0
        # temp_list = []
        for u in tqdm(self.network.nodes(), "update parameters"):
            new_data_num = new_node_matrix[u]
            vector_num += new_data_num
            if new_data_num != 0:
                node_num += 1
                new_data_mean = np.mean(self.online_data[label_matrix[u]], axis=0, keepdims=True)


                if type(self.network.nodes[u]['vector']) is np.ndarray:
                    self.network.nodes[u]['cov'], self.network.nodes[u]['vector'] = self.incre_par(
                        self.network.nodes[u]['nodes_num'], new_data_num,
                        self.network.nodes[u]['vector'].reshape(1, -1),
                        new_data_mean, self.network.nodes[u]['cov'],
                        np.cov(self.online_data[label_matrix[u]],
                               rowvar=False))
                    self.network.nodes[u]['cov_inv'] = np.linalg.inv(self.network.nodes[u]['cov'] + 0.01 * I)
                    self.network.nodes[u]['nodes_num'] += new_data_num
                else:
                    nodes_to_remove.append(u)

        self.del_nul_nodes(nodes_to_remove)
        self.prune_connections()
        self.del_nodes()

        center_list = []
        node_idx = []
        vec_num = []
        new_num = []
        for u in self.network.nodes():
            if u < self.init_node:
                vec_num.append(self.network.nodes[u]['threshold'])
            else:
                if self.network.nodes[u]['threshold'] != float("-inf"):
                    new_num.append(self.network.nodes[u]['threshold'])
            center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
            node_idx.append(u)
        print("total neural node now:\t", len(center_list))
        self.node_list = np.asarray(node_idx)
        self.centeroid = np.concatenate(center_list)
        vec_num.sort(reverse=True)

        print("update node_num:\t", node_num, "new vec num:\t", vector_num)
        print("total neural node:\t", self.units_created)
        print("Out S1 and S2:\t", count, "Out S1 in S2", cout1)

    def online_gpu_cal_dist(self, data):
        observation = torch.from_numpy(data)
        B, L = observation.shape
        centeroid = torch.from_numpy(self.centeroid)
        centeroid = centeroid.t().unsqueeze(0).cuda()
        dist_list = []
        index_list = []
        batch_size = 500
        for d_idx in range(B // batch_size + 1):
            if d_idx == B // batch_size:
                test_batch = observation[d_idx * batch_size:]
                test_batch = test_batch.cuda()
            else:
                test_batch = observation[d_idx * batch_size:d_idx * batch_size + batch_size]
                test_batch = test_batch.cuda()
            dist_matrix = torch.pairwise_distance(test_batch.unsqueeze(-1),
                                                  centeroid)
            topk_values, topk_indexes = torch.topk(dist_matrix, k=2, dim=1, largest=False, sorted=True)
            dist_list.append(torch.sqrt(topk_values).cpu())
            index_list.append(topk_indexes.cpu())
        topk_values = torch.cat(dist_list)
        topk_indexes = torch.cat(index_list)
        return topk_values, topk_indexes

    def update_threshold(self):
        topk_values, _ = self.online_gpu_cal_dist(self.centeroid)
        topk_values = topk_values.numpy()

        sta_neighbor = []
        sta_th = []
        sta_dist = []
        for i in tqdm(range(len(self.centeroid)), "| update threshold |"):
            u = self.node_list[i]
            sta_dist.append(topk_values[i][1])
            if self.network.degree(u) == 0:
                dist_1 = topk_values[i][1]
                self.network.nodes[u]['threshold'] = dist_1
            else:
                neighbors = list(self.network.neighbors(u))
                sta_neighbor.append(len(neighbors))
                dist_u = []
                for neighbor in neighbors:
                    neighbor_dist = np.sqrt(np.sum(
                        (self.network.nodes[u]['vector'] - self.network.nodes[neighbor]['vector']) ** 2))
                    dist_u.append(neighbor_dist)
                # self.network.nodes[u]['threshold'] = np.mean(dist_u)
                self.network.nodes[u]['threshold'] = np.median(dist_u)
            sta_th.append(self.network.nodes[u]['threshold'])
        print("nodes have neighbor: \t", len(sta_neighbor), "average neighbor:\t", np.mean(sta_neighbor))
        print("average threshold:\t", np.mean(sta_th))
        print("average distance:\t", np.mean(sta_dist))


    def cluster_label(self):
        label_matrix = [[] for i in range(self.units_created)]
        node_idx = []
        center_list = []
        for u in self.network.nodes():
            node_idx.append(u)
            center_list.append(self.network.nodes[u]['vector'].reshape(1, -1))
        self.centeroid = np.concatenate(center_list)
        self.node_list = np.asarray(node_idx)
        C = self.data.shape[1]
        I = np.identity(C)
        dist_matrix = self.gpu_cal_dist(self.data)
        topk_values, topk_indexes = torch.topk(dist_matrix, k=1, dim=1, largest=False, sorted=True)
        topk_indexes = topk_indexes.numpy()
        topk_values = topk_values.numpy()
        for i in tqdm(range(len(self.data)), "| assign label |"):
            s_1 = node_idx[topk_indexes[i][0]]
            dist_1 = topk_values[i][0]
            # 更新阈值
            if dist_1 > self.network.nodes[s_1]['threshold']:
                self.network.nodes[s_1]['threshold'] = dist_1
            label_matrix[s_1].append(i)
        nodes_to_remove = []
        for u in self.network.nodes():
            if (type(self.network.nodes[u]['vector']) is np.ndarray) and (len(self.data[label_matrix[u]]) > 1):
                self.network.nodes[u]['cov'] = np.cov(self.data[label_matrix[u]], rowvar=False)
                self.network.nodes[u]['cov_inv'] = np.linalg.inv(self.network.nodes[u]['cov'] + 0.01 * I)
            else:
                nodes_to_remove.append(u)
        self.del_nul_nodes(nodes_to_remove)


    def prune_connections(self):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > self.age_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

    def del_nodes(self):
        # 删除孤立神经元
        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                if u > self.init_node:
                    nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)
        print("remove isolate nodes\t", len(nodes_to_remove))

    def del_nul_nodes(self, remove_list):
        index = []
        for u in remove_list:
            self.network.remove_node(u)
            index.append(np.argwhere(self.node_list == u).squeeze())
        self.centeroid = np.delete(self.centeroid, index, axis=0)
        self.node_list = np.delete(self.node_list, index, axis=0)

    def incre_par(self, old_num, new_num, old_mean, new_mean, old_cov, new_cov):
        glob_mean = (old_num * old_mean + new_num * new_mean) / (old_num + new_num)
        a1 = (old_num - 1) * old_cov + np.matmul((glob_mean - old_mean).T, (glob_mean - old_mean)) * old_num
        a2 = (new_num - 1) * new_cov + np.matmul((glob_mean - new_mean).T, (glob_mean - new_mean)) * new_num
        return (a1 + a2) / (old_num + new_num - 1 + 1e-6), glob_mean
