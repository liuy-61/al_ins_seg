import torch
import numpy as np
from liuy.utils.local_cofig import OUTPUT_DIR
import os
from scipy.spatial import distance_matrix
from liuy.utils.local_config import VAE_feature_path

import pandas as pd
import copy


def cross_correlation(feature1, feature2):
    """

    :param feature1: tensor: (M,C,output_size,output_size)
     M is total number of mask features' corresponding to a image
    :param feature2: tensor: (N,C,output_size,output_size)
    M is total number of mask features' corresponding to a image
    :return: a scale represent the similarity between feature1 and feature2
    """

    """
        compute the cross correlation (M * N) times,compute the every mask features between tow images
    """
    M = feature1.shape[0]
    N = feature2.shape[0]
    sum_similarity = 0
    for i in range(M):
        for j in range(N):
            simlilarity = feature1[i] * feature2[j]
            simlilarity = torch.sum(simlilarity)
            sum_similarity += simlilarity
    return sum_similarity / (M * N)


def mapfuc(feature):
    feature = feature[1:-1]
    feature = feature.split(',')
    feature = [float(x) for x in feature]
    return feature


class CoreSetSampler:
    def __init__(self, sampler_name, project_id, whole_image_id_list):
        """
        :param sampler_name: custom
        :param project_id : use to find the path saved the mask feature
        :param  whole_image_id_list
        """
        self.sampler_name = sampler_name
        self.project_id = project_id
        self.whole_image_id_list = whole_image_id_list

    def greedy_k_center(self, labeled, unlabeled, amount):
        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount - 1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
        return np.array(greedy_indices)

    def select_batch(self, n_sample, already_selected, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param kwargs:
        :return: list of image_id you selected this batch
        """
        latents = []
        df_feature = pd.read_csv(VAE_feature_path)
        feature_str = df_feature["feature"].values
        for feature in feature_str:
            latents.append(mapfuc(feature))
        latents = np.array(latents)

        all_img_list = copy.deepcopy(self.whole_image_id_list)
        labeled_list = already_selected
        unlabeled_list = np.setdiff1d(all_img_list, labeled_list)
        # 由于greedy_k_center是通过1，2，3的索引访问latens数组，最终返回的也是索引。而不是img_id，所以我们需要对img_id和索引index做一个映射的字典
        id2index = {}
        index2id = {}
        length = len(all_img_list)
        for i in range(length):
            id2index[all_img_list[i]] = i
            index2id[i] = all_img_list[i]

        # 再将list中的img_id转为索引index，例如第一张图像的img_id是36，转成索引为0；第二张图像的img_id是49，转成索引为1；
        for i in range(len(labeled_list)):
            labeled_list[i] = id2index[labeled_list[i]]
        for i in range(len(unlabeled_list)):
            unlabeled_list[i] = id2index[unlabeled_list[i]]

        sample_list = self.greedy_k_center(
            labeled=latents[labeled_list, :],
            unlabeled=latents[unlabeled_list, :],
            amount=n_sample
        )

        for i in range(len(sample_list)):
            sample_list[i] = index2id[sample_list[i]]

        return list(sample_list)


if __name__ == '__main__':
    pass
