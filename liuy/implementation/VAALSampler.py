import random
import numpy as np
from scipy.spatial import distance_matrix
from liuy.utils.local_config import VAE_feature_path

import pandas as pd
import copy


def read_img_list(path):
    with open(path, 'r') as f:
        result = list(f.readlines())
        img_str = result[0]
        img_str = img_str.split(' ')
        img_list = [int(item) for item in img_str]
        print("--load {} samples".format(len(img_list)))
        return img_list

def mapfuc(feature):
    feature = feature[1:-1]
    feature = feature.split(',')
    feature = [float(x) for x in feature]
    return feature


class VAALSampler:
    def __init__(self, sampler_name, project_id, whole_image_id_list):
        """
        :param sampler_name: custom
        :param project_id : use to find the path saved the mask feature
        :param  whole_image_id_list
        """
        self.sampler_name = sampler_name
        self.project_id = project_id
        self.whole_image_id_list = whole_image_id_list

    def select_batch(self, n_sample, already_selected, **kwargs):
        latents = []
        df_feature = pd.read_csv(VAE_feature_path)
        feature_str = df_feature["feature"].values
        for feature in feature_str:
            latents.append(mapfuc(feature))
        latents = np.array(latents)

        all_img_list = copy.deepcopy(self.whole_image_id_list)
        labeled_list = copy.deepcopy(already_selected)
        unlabeled_list = np.setdiff1d(all_img_list, labeled_list)
        for i in labeled_list:
            for j in unlabeled_list:
                if i == j:
                    print("i == j1 {}".format(i))

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
            labeled_idx=labeled_list,
            unlabeled_idx=unlabeled_list,
            representation=latents,
            amount=n_sample
        )

        for i in range(len(sample_list)):
            sample_list[i] = index2id[sample_list[i]]
        for i in range(len(labeled_list)):
            labeled_list[i] = index2id[labeled_list[i]]
        for i in range(len(unlabeled_list)):
            unlabeled_list[i] = index2id[unlabeled_list[i]]

        return list(sample_list)


if __name__ == '__main__':
    random.seed(61)
    whole_image_id_list = read_img_list("/home/muyun99/Desktop/coco_output/selected_img_list/coreset/100")

    already_selected = random.sample(whole_image_id_list, 100)

    sampler = VAALSampler(
        sampler_name='VAAL',
        project_id="VAAL_test",
        whole_image_id_list=whole_image_id_list
    )
    sample_list = sampler.select_batch(n_sample=1000, already_selected=already_selected)