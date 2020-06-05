import pickle
import random

import torch
import torch.nn as nn

from detectron2.engine import default_argument_parser
from liuy.utils.local_cofig import OUTPUT_DIR, VAE_feature_path ,score_list_path
import os
from torch.utils.tensorboard import SummaryWriter

from liuy.utils.reg_dataset import register_coco_instances_from_selected_image_files
from liuy.utils.torch_utils import select_device
from liuy.utils.K_means import handle_feature_cvs
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.utils.local_cofig import coco_data,debug_data
from liuy.utils.create_curve import generate_base_model, load_base_model, read_selected_image_id
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.RegressorSampler import RegressionSampler
"""
    a regression model is trained by feature and the score of this feature
    after training the model can predict the feature's score given a feature
"""
LR = 0.0001
Epoch = 1

class Regression(torch.nn.Module):
    """
        this version of Regression use the mask_feature as data
    """
    def __init__(self, regression_id, in_dim=64,n_hidden1=32,n_hidden2=16,n_hidden3=8,n_hidden4=4,out_dim=1):
        super(Regression, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, n_hidden3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden3, n_hidden4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden4, out_dim), nn.ReLU(True))

        self.dir = os.path.join(OUTPUT_DIR, regression_id)

    def forward(self, feature):
        """

        :param feature: tensor: (1, 64)
        :return: score
        """
        x = self.layer1(feature)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def get_predict(self, feature_path):
        """

        :param feature_path:
        :return: pre_list :list[dict] {'image_id': int, 'score':tensor}
        """
        device = select_device()
        pre_list = []
        feature = handle_feature_cvs(feature_path)

        feature_list = feature[:100, :-1]
        feature_id = feature[:, -1]

        feature_tensor = []
        for item in feature_list:
            tensor = torch.from_numpy(item.reshape(1, -1))
            feature_tensor.append(tensor)



        data_tensor = feature_tensor[0]
        for i, item in enumerate(feature_tensor):
            if i != 0:
                data_tensor = torch.cat([data_tensor, item], dim=0)

        data_tensor = torch.tensor(data_tensor, dtype=torch.float32)

        torch_dataset = Data.TensorDataset(data_tensor)
        data_loader = DataLoader(
            dataset=torch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
        for i, data in enumerate(data_loader):
            data[0] = data[0].to(device)
            score = self.forward(data[0])
            pre_dict = {'score': score, 'image_id': int(feature_id[i])}
            pre_list.append(pre_dict)

        return pre_list





def train_regre(regression_model, feature_path, score_list_path, lr=0.0001, Epoch=1):
    """

    :param regression_model:
    :param feature_path: the path to the data(feature)
    :param score_list_path:  the path to the label(score)
    :param Lr:
    :param Epoch:
    :return:
    """
    device = select_device()
    regression_model.train()
    regression_model = regression_model.to(device)

    if not os.path.exists(regression_model.dir):
        os.makedirs(regression_model.dir)
    writer = SummaryWriter(log_dir=regression_model.dir)

    optimizer = torch.optim.Adam(regression_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    feature_score_list = contact_feature_score(feature_path=feature_path,
                                               score_list_path=score_list_path)

    # list[dict] dict:{'image_id': int, 'score': tensor, 'feature': tensor}
    # build data loader

    data_tensor = feature_score_list[0]['feature']
    target_tensor = feature_score_list[0]['score']

    for i, item in enumerate(feature_score_list):
        if i != 0:
            data_tensor = torch.cat([data_tensor, item['feature']], dim=0)
            target_tensor = torch.cat([target_tensor, item['score']], dim=0)

    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

    torch_dataset = Data.TensorDataset(data_tensor, target_tensor)
    data_loader = DataLoader(
        dataset=torch_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
    )

    for epoch in range(Epoch):
        for step, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            output = regression_model(data)

            loss = loss_func(target, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss :{}'.format(loss))
            writer.add_scalar('train_loss', loss)

    save_regression_model(regression_model)


def save_regression_model(regression_model):
    # self.cfg.OUTPUT_DIR
    # with open(os.path.join(self.cfg.OUTPUT_DIR, self.project_id + "_model" + ".pkl"), 'wb') as f:
    # after training save the regression model
    if not os.path.exists(regression_model.dir):
        os.makedirs(regression_model.dir)

    with open(os.path.join(regression_model.dir, "regression_model.pkl"), 'wb') as f:
        pickle.dump(regression_model, f)


def load_regression_model(regression_id):
    dir = os.path.join(OUTPUT_DIR, regression_id)
    if os.path.exists(os.path.join(dir, 'regression_model.pkl')):
        with open(os.path.join(dir, 'regression_model.pkl'), 'rb') as f:
            model = pickle.load(f)
            return model
    else:
        return None


def min_max_norm(score_list):
    tmp = [x['score'] for x in score_list]
    minum = min(tmp)
    maxum = max(tmp)
    tmp = [(x-minum)/(maxum-minum) for x in tmp]
    for i, dict in enumerate(score_list):
        dict['score'] = tmp[i]


def read_mask_feature(detail_output_dir, serial_number):
    if os.path.exists(detail_output_dir + '/' + 'unselected' + str(serial_number) + '.pkl'):
        with open(detail_output_dir + '/' + 'unselected' + str(serial_number) + '.pkl', 'rb') as f:
            return pickle.load(f)


def contact_feature_score(feature_path, score_list_path):
    """
    :param feature_path:
    :param score_list_path:
    score_list : list[dict] dict:{'image_id': int, 'score': float}
    feature_list : nd array M * N  M is the number of feature,
                each feature has N-1 dimension and the N th
                is the feature id
    :return: list[dict] dict:{'image_id': int, 'score': tensor (1,1), 'feature': tensor (1,64)}
    """
    feature_list = handle_feature_cvs(feature_path)

    with open(score_list_path, 'rb') as f:
        score_list = pickle.load(f)

    feature_score_list = []
    feature_id_list = feature_list[:, -1].tolist()
    for item in score_list:
        image_id = item['image_id']
        feature_score = {'image_id': image_id, 'score': torch.tensor(item['score']).reshape(1, 1)}
        feature = feature_list[feature_id_list.index(image_id), :-1]
        feature = torch.from_numpy(feature.reshape(1, -1))
        feature_score['feature'] = feature
        feature_score_list.append(feature_score)

    return feature_score_list


def generate_one_curve(
        whole_image_id,
        coco_data,
        sampler,
        ins_seg_model,
        seed_batch,
        batch_size
):
    # initialize the quantity relationship
    whole_train_size = len(whole_image_id)
    if seed_batch < 1:
        seed_batch = int(seed_batch * whole_train_size)
    if batch_size < 1:
        batch_size = int(batch_size * whole_train_size)

    # initally, seed_batch pieces of image were selected randomly
    selected_image_id = random.sample(whole_image_id, seed_batch)
    # register data set and build data loader
    register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_id)
    data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image')

    n_batches = int(np.ceil(((whole_train_size - seed_batch) * 1 / batch_size))) + 1
    for n in range(n_batches):
        # check the size in this iter
        n_train_size = seed_batch + min((whole_train_size - seed_batch), n * batch_size)
        print('{} data ponints for training in iter{}'.format(n_train_size, n))
        assert n_train_size == len(selected_image_id)


        ins_seg_model.save_selected_image_id(selected_image_id)

        ins_seg_model.fit_on_subset(data_loader_from_selected_image_files)

        # get the losses for loss_sampler
        losses = ins_seg_model.compute_loss(json_file=coco_data[0]['json_file'],
                                            image_root=coco_data[0]['image_root'],)


        n_sample = min(batch_size, whole_train_size - len(selected_image_id))
        new_batch = sampler.select_batch(n_sample, already_selected=selected_image_id, losses=losses, loss_decrease=False)
        selected_image_id.extend(new_batch)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))

        # register dataset and build data loader
        register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                          json_file=coco_data[0]['json_file'],
                                                          image_root=coco_data[0]['image_root'],
                                                          selected_image_files=selected_image_id)
        data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
            'coco_from_selected_image')
        assert len(new_batch) == n_sample

        # reset model if
        ins_seg_model.reset_model()



if __name__ == '__main__':
    """
        train some base model use separately 20% data 30% data ......100% data

    """
    coco_data = debug_data

    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='Base', coco_data=coco_data, resume_or_load=True)

    data_loader = seg_model.trainer.data_loader
    whole_image_id = [item['image_id'] for item in data_loader.dataset._dataset._lst]

    # generate_base_model(whole_image_id=whole_image_id,
    #                     coco_data=coco_data,
    #                     ins_seg_model=seg_model,
    #                     seed_batch=0.2,
    #                     batch_size=0.1)
    """
        load the trained base models, and use base models to fit_on_single_data get score_list
        the score_list will be saved as OUTPUT_DIR/file/score_list
    """
    # serial_num = 0
    #
    # sampler = CoCoRandomSampler(sampler_name='random',
    #                             whole_image_id=whole_image_id)
    #
    # while True:
    #     # the base model is generalized rcnn
    #     base_model = load_base_model(project_id='Base',
    #                                  serial_num=serial_num)
    #
    #     selected_image_id = read_selected_image_id(project_id='Base',
    #                                                serial_num=serial_num)
    #     serial_num += 1
    #
    #     if selected_image_id is not None:
    #         seg_model.set_model(base_model)
    #
    #         # for last base model we do not fit_on_single_data
    #         if len(selected_image_id) == len(whole_image_id):
    #             break
    #         else:
    #             n_sample = int(0.005*len(whole_image_id))
    #             if n_sample > len(whole_image_id)-len(selected_image_id):
    #                 n_sample = len(whole_image_id)-len(selected_image_id)
    #             new_batch = sampler.select_batch(n_sample=n_sample,
    #                                              already_selected=selected_image_id)
    #             selected_image_id.extend(selected_image_id)
    #
    #             seg_model.fit_on_single_data(new_batch)
    #
    #
    #     else:
    #         break
    #

    """
        load VAE model, get VAE_feature, save the VAE feature
    """


    """
        train the regression model
    """
    regression_model = Regression(regression_id='regression')

    train_regre(regression_model,
          feature_path=VAE_feature_path,
          score_list_path=score_list_path,
          Epoch=4)


    # del seg_model

    predict_score = regression_model.get_predict(VAE_feature_path)

    ins_seg_model = CoCoSegModel(args, project_id='reg', coco_data=coco_data, resume_or_load=True)

    whole_image_id = [item['image_id'] for item in data_loader.dataset._dataset._lst]

    sampler = RegressionSampler(regression_model=regression_model,
                                sampler_name='regression',
                                feature_score=predict_score)

    generate_one_curve(whole_image_id=whole_image_id,
                       coco_data=coco_data,
                       sampler=sampler,
                       ins_seg_model=ins_seg_model,
                       seed_batch=0.2,
                       batch_size=0.1)






