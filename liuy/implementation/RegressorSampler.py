import pickle
from operator import itemgetter
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from liuy.utils.local_cofig import OUTPUT_DIR
import os
from torch.utils.tensorboard import SummaryWriter
from liuy.utils.torch_utils import select_device
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.utils.reg_dataset import register_coco_instances_from_selected_image_files
from liuy.utils.local_cofig import coco_data
from detectron2.engine import default_argument_parser
# from liuy.utils.local_cofig import OUTPUT_DIR

LR = 0.0001
Epoch = 1
Logdir = os.path.join(OUTPUT_DIR, 'regression')




class RegressionSampler():
    def __init__(self, sampler_name, regression_model, feature_score):
        """

        :param sampler_name:
        :param regression_model:
        :param feature_score: :list[dict] {'image_id': int, 'score':tensor}
        """
        self.feature_score = feature_score
        self.sampler_name = sampler_name
        self.regression_model = regression_model

    def select_batch(self,  n_sample, already_selected, ):
        """

        :param feature_path:
        :param self:
        :param regression: the model to predict the mask feature score
        :param n_sample:
        :param already_selected:
        :return:
        """
        device = select_device()
        sample = []
        # score_dict: list of dict, dict {'image_id': int, 'score':tensor}

        # sort the score_dict, and select the batch with highest score
        sorted_score = sorted(self.feature_score, key=itemgetter("score"), reverse=True)
        cnt = 0
        i = 0
        while cnt < n_sample:
            if sorted_score[i] not in already_selected and sorted_score[i] not in sample:
                sample.append(sorted_score[i]['image_id'])
                cnt += 1
            i += 1

        assert len(sample) == n_sample
        return sample



if __name__ == '__main__':
    pass





