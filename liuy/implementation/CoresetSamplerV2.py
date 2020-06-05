import torch
import numpy as np
from liuy.utils.local_config import OUTPUT_DIR
from liuy.utils.save_mask_feature import read_mask_feature
import os


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


class CoreSetSampler:
    def __init__(self, sampler_name, project_id, whole_image_id_list):
        """
        :param sampler_name: custom
        :param project_id : use to find the path saved the mask feature
        :param  whole_image_id_list
        """
        self.sampler_name = sampler_name
        self.mask_feature_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id)
        self.project_id = project_id
        self.whole_image_id_list =whole_image_id_list


    def select_batch(self, n_sample, already_selected=None, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param kwargs:
        :return: list of image_id you selected this batch
        """

        return self.greedy_k(amount=n_sample)

    def greedy_k(self, amount):
        """

        :param amount:
        :return: greedy_indices, a list of image_id, image_id :int
        """
        greedy_indices = []
        # similarity_matrix, assume there are M unselected mask_features and N selected mask_features,
        # the shape of similarity_matrix is (M,N),
        # meaning the similarity between M unselected mask_features and N selected mask_features
        similarity_matrix = np.array([])

        # dim2image_id_list maps the similarity_matrix dim 0 to image id
        dim2image_id_list = []

        # read all unselected_mask_feature
        # unselected_serial_number and selected_serial_number are used to read the mask features
        unselected_serial_number = 0
        while True:
            unselected_mask_feature = read_mask_feature(project_id=self.project_id,
                                                        serial_number=unselected_serial_number,
                                                        selected_or_not=None)
            if unselected_mask_feature:
                for un in unselected_mask_feature:
                    """
                    """
                    dim2image_id_list.append(un['image_id'])
                    similarity_array = np.array([])

                    # read all selected_mask_feature
                    selected_serial_number = 0
                    while True:
                        selected_mask_feature = read_mask_feature(project_id=self.project_id,
                                                                  serial_number=selected_serial_number,
                                                                  selected_or_not=True)

                        if selected_mask_feature:
                            for sel in selected_mask_feature:
                                similarity = cross_correlation(un['feature_tensor'], sel['feature_tensor'])
                                similarity_array = np.append(similarity_array, similarity)
                                # TODO Handel the similarity,

                            selected_serial_number += 1
                        else:
                            break

                    if similarity_matrix.size == 0:
                        similarity_matrix = similarity_array
                    else:
                        similarity_matrix = np.vstack((similarity_matrix, similarity_array))

                unselected_serial_number += 1
            else:
                break
        # max_similarity, each unselected image's mask features' similarity to the whole selected images
        max_similarity = np.max(similarity_matrix, axis=1).reshape(-1, 1)

        least_similarity = np.argmin(max_similarity)
        greedy_indices.append(dim2image_id_list[least_similarity])
        max_similarity[least_similarity, 0] = float('inf')

        for i in range(amount - 1):
            last_selected = {}
            # continue to read unselected mask_feature until find last_selected
            unselected_serial_number = 0
            while True:
                unselected_mask_feature = read_mask_feature(project_id=self.project_id,
                                                            serial_number=unselected_serial_number,
                                                            selected_or_not=None)
                if unselected_mask_feature and last_selected is None:
                    for un in unselected_mask_feature:
                        if un['image_id'] == greedy_indices[-1]:
                            last_selected = un
                            break
                    unselected_serial_number += 1
                else:
                    break

            # compute similarity between every unselected mask feature and last selected mask feature
            # save the similarity to similarity_array
            similarity_array = np.array([])
            unselected_serial_number = 0
            while True:
                unselected_mask_feature = read_mask_feature(project_id=self.project_id,
                                                            serial_number=unselected_serial_number,
                                                            selected_or_not=None)
                if unselected_mask_feature:
                    for un in unselected_mask_feature:
                        similarity = cross_correlation(un['feature_tensor'], last_selected['feature_tensor'])
                        similarity_array = np.append(similarity_array, similarity)
                    unselected_serial_number += 1
                else:
                    break

            # h stack the max_similarity and similarity_array, and update the max_similarity
            similarity_array = similarity_array.reshape(-1, 1)
            max_similarity = np.hstack((max_similarity, similarity_array))
            max_similarity = np.max(max_similarity, axis=1).reshape(-1, 1)

            least_similarity = np.argmin(max_similarity)

            greedy_indices.append(dim2image_id_list[least_similarity])
            max_similarity[least_similarity, 0] = float('inf')


    if __name__ == '__main__':
        a = [1,2,3]

        print ()

    debug = 1
