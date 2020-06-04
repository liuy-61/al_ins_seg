import torch
import numpy as np
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
    return sum_similarity / (M*N)


class CoreSetSampler:
    def __init__(self, sampler_name, mask_feature):
        """
        :param sampler_name: custom
        :param mask_feature:  a list of dict, dict :{'image_id':int, 'feature_tensor':tensor}
        the  tensor's shape is (M, C, out_put_size, out_put_size) M is total number of the mask
        corresponding  image from batched_inputs ( the batch size is 1 )
        turn mask_feature from a list to numpy nd array
        """
        self.sampler_name = sampler_name
        self.mask_feature = np.array(mask_feature)
        whole_image_id = [item['image_id'] for item in mask_feature]
        self.whole_image_id = np.array(whole_image_id)


    def select_batch(self, n_sample, already_selected, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param already_selected: list of image_id which has been already selected
        turn it from list to a np.array
        :param kwargs:
        :return: list of image_id you selected this batch
        """
        already_selected = np.array(already_selected)

        not_selected = self.get_not_selected(already_selected)
        labeled = np.in1d(self.whole_image_id, already_selected)
        labeled = self.mask_feature[labeled]
        un_labeled = np.in1d(self.whole_image_id, not_selected)
        un_labeled = self.mask_feature[un_labeled]
        return self.greedy_k(n_sampel=n_sample, labeled=labeled, unlabeled=un_labeled)

    def greedy_k(self, n_sampel, labeled, unlabeled):
        """

        :param n_sampel: amount of this batch to select
        :param labeled:  np.array of dict, dict :{'image_id':int, 'feature_tensor':tensor}
        :param unlabeled: np.array of dict, dict :{'image_id':int, 'feature_tensor':tensor}
        :return: greedy_indices: a list of image_id you selected this batch
        """
        greedy_indices = []
        similarity_list = []
        """ a list of dict,dict : {'image_id': int, 'tensor_feature': tensor ,'similarity': scale} , 
            meaning each unlabeled image's tensor_feature and the similarity to whole labeled image
        """
        # compute similarity between  every unlabeled image and whole labeled images set
        for un in unlabeled:
            similarity = 0
            for la in labeled:
                correlation = cross_correlation(un['feature_tensor'], la['feature_tensor'])
                if similarity < correlation:
                    similarity = correlation
            similarity_dict = {'image_id': un['image_id'], 'feature_tensor': un['feature_tensor'], 'similarity': similarity}
            similarity_list.append(similarity_dict)

        # get the least similar unlabeled image to the whole labeled images set,
        # add the least similar unlabeled image to greedy_indices,
        # remove the least similar unlabeled image from unlabeled images set
        least_similar = similarity_list[0]
        for item in similarity_list:
            if least_similar['similarity'] > item['similarity']:
                least_similar = item

        greedy_indices.append(least_similar['image_id'])

        similarity_list.remove(least_similar)

        unlabeled = unlabeled.tolist()
        unlabeled.remove({'image_id': least_similar['image_id'], 'feature_tensor': least_similar['feature_tensor']})
        unlabeled = np.array(unlabeled)

        for i in range(n_sampel-1):
            for index, un in enumerate(unlabeled):
                # compute the similarity between the every unlabeled image
                # and the least similar unlabeled image last got, if the similarity
                # is bigger than before update the similarity
                correlation = cross_correlation(un['feature_tensor'], least_similar['feature_tensor'])
                if similarity_list[index]['similarity'] < correlation:
                    similarity_list[index]['similarity'] = correlation

            # get the least similar unlabeled image to the whole labeled images set,
            # add the least similar unlabeled image to greedy_indices,
            # remove the least similar unlabeled image from unlabeled images set
            least_similar = similarity_list[0]
            for item in similarity_list:
                if least_similar['similarity'] > item['similarity']:
                    least_similar = item

            greedy_indices.append(least_similar['image_id'])
            print("select {} images, still need {} images".format(i+2, n_sampel-i-2))
            similarity_list.remove(least_similar)

            unlabeled = unlabeled.tolist()
            unlabeled.remove({'image_id': least_similar['image_id'], 'feature_tensor': least_similar['feature_tensor']})
            unlabeled = np.array(unlabeled)


        assert len(greedy_indices) == n_sampel
        return greedy_indices

    def get_not_selected(self, already_selected):
        """

        :param already_selected: np.array  of image_id,image_id : int,image_id indicate the image which has been selected
        :return:not_selected:  np.array  image_id,image_id : int,image_id indicate the image which has not been selected
        """
        not_selected = self.whole_image_id[np.logical_not(np.in1d(self.whole_image_id, already_selected))]
        return not_selected

