"""
随机创建K个center
当任意一个点的簇分配结果发生改变时
    对数据集特征的每一个点
        对每一个center
            计算center到特征点的距离
        将特征点分配到最近的簇
    对每一个簇,计算簇中所有特征点的均值并将其作为新的center
直到簇不再发生变化或者达到最大迭代数
"""

from liuy.utils.local_cofig import feature_path
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt

def k_means(feature_path, k):
    """

    :param feature_path:
    :param K:
    :return: a nd array, m * 2, m means number of feature, the first column is the
    image id, and the second is the class id
    """
    feature = handle_feature_cvs(feature_path)

    # m is number of feature , n-1 is each feature's dimension,
    # nth dimension is image_id to the feature
    m, n = feature.shape

    # randomly create k centroids,
    k_centroids = randCent(feature, k)

    # initialize container, clusterAssment
    # the first column save the distance between each feature point to cluster centroid
    # the second column  each feature point belong to which cluster centroid this iter
    # the third  column  each feature point belong to which cluster centroid last iter
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf
    clusterAssment[:, 1:] = -1


    result_sets = np.hstack((feature, clusterAssment))

    clusterChanged = True
    iteration_cnt = 0
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            dist = distance_matrix(feature[i, :n - 1].reshape((1, -1)), k_centroids)
            result_sets[i, n] = np.min(dist)
            result_sets[i, n + 1] = np.argmin(dist)
        clusterChanged = not (result_sets[:, -1] == result_sets[:, -2]).all()

        if clusterChanged:
            iteration_cnt += 1
            print("iteration num is {}".format(iteration_cnt))
            debug = 1
            # update k_centroids
            result_pd = pd.DataFrame(result_sets)
            cent_df = result_pd.groupby(n+1).mean()
            k_centroids = cent_df.iloc[:, :n-1].values
            result_sets[:, -1] = result_sets[:, -2]

    image_id = result_sets[:, n]
    class_id = result_sets[:, -2]
    image2class = np.hstack((image_id, class_id))
    return image2class
    debug = 1




def randCent(feature, k):
    """
    :param feature:  ndarray M * N  M is the number of feature,
    each feature has N-1 dimension and the N th
    is the feature id
    :param K: the number of centroid to build
    :return:
    randomly build k centroid for feature
    """
    n = feature.shape[1]

    feature_value = feature[:, :-1]
    data_min = np.min(feature_value)
    data_max = np.max(feature_value)

    data_cent = np.random.uniform(data_min, data_max, (k, n-1))
    return data_cent

def mapfuc(feature_str):
    feature = feature_str[1:-1]
    feature = feature.split(',')
    feature = [float(x) for x in feature]
    return feature

def handle_feature_cvs(feature_path):
    """

    :param feature_path: the feature cvs file  path
    :return: ndarray M * N  M is the number of feature,
    each feature has N-1 dimension and the N th
    is the feature id
    """
    feature_container = pd.read_csv(feature_path)
    feature_id = feature_container['image_path'].values
    feature_id = np.array(feature_id)
    feature_id = feature_id.reshape((feature_id.shape[0],1))

    feature_str = feature_container['feature'].values
    feature = []
    for item in feature_str:
        feature.append(mapfuc(item))
    feature = np.array(feature)

    features = np.hstack((feature, feature_id))
    return features
    debug = 1


if __name__ == '__main__':
   k_means(feature_path=feature_path, k=10)