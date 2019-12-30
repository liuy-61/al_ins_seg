from detectron2.engine import default_argument_parser
from liuy.Interface.BaseAl import BaseAl
from liuy.implementation.InsSegModel import InsSegModel
from liuy.implementation.RandomSampler import RandomSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files
# class AlModel(BaseAl):
#     def __init__(self, segm_model, sampler, batch_size, warmstart_size):
#         """
#         :param segm_model: model used to score the samplers.  Expects fit and predict
#         methods to be implemented.
#         :param sampler: sampling class from sampling_methods, assumes reference
#         passed in and sampler not yet instantiated.
#         :param batch_size: float or int.  float indicates batch size as a percent of
#       training data
#         :param warmstart_size: float or int.  float indicates percentage of train data
#       to use for initial model
#         """
#         super(AlModel).__init__(segm_model, sampler)
#         self.batch_size = batch_size
#         self.warmstart_size = warmstart_size
#
#     def select_batch(self, n_sample, already_selected,
#                      **kwargs):
#         kwargs['n_sample'] = n_sample
#         kwargs['already_selected'] = already_selected
#         return self.sampler.select_batch(**kwargs)
def generate_one_curve(data_loader,
                       sampler,
                       ins_seg_model,
                       seed_batch,
                       batch_size,
                       ):
    """
    
    :param data_loader:      the data_loader contains all training data , we use the sampler select data（image） from it.
    :param sampler:          active learning sampler
    :param ins_seg_model:    model used to score the samplers.  Expects fit and test methods to be implemented.
    :param seed_batch: float（from 0 to 1）   float indicates percentage of train data to use for initial model
    :param batch_size: float （from 0 to 1）   float indicates batch size as a percent of training data ,
    we use sampler select batch_size peaces of data （image）
    :return: 
    """
    # def select_batch(sampler, n_sample, already_selcted, **kwargs):
    #     """
    # 
    #     :param sampler:         active learning sampler
    #     :param n_sample:        we select n_sample pieces of data（image）
    #     :param already_selcted: Data （image）that has been selected before
    #     :param kwargs:
    #     :return:
    #     """
    #     kwargs['n_sample'] = n_sample
    #     kwargs['already_selected'] = already_selcted
    #     batch = sampler.select_batch(**kwargs)
    #     return batch
    
    # get all the image files from the data_loader
    image_files_list = []
    list = data_loader.dataset._dataset._lst
    for item in list:
        image_files_list.append(item['file_name'])

    # The size of the entire training set
    train_size = len(image_files_list)
    # transform seed_batch and batch_size from float which indicate percentage of entire training set to int
    seed_batch = int(seed_batch * train_size)
    batch_size = int(batch_size * train_size)

    # We recorded the results of the model training and testing after each data sampling
    results = {}
    data_sizes =[]
    mious = []
    
    # initally, seed_batch pieces of image were selected randomly
    seleted_image_files = random.sample(image_files_list, seed_batch)

    register_a_cityscapes_from_selected_image_files(image_dir='/media/tangyp/Data/cityscape/leftImg8bit/train',
                                                    gt_dir='/media/tangyp/Data/cityscape/gtFine/train',
                                                    selected_image_files=seleted_image_files,
                                                    dataset_name='dataset_from_seleted_iamge_files'
                                                    )
    data_loader_from_selected_iamge_files = ins_seg_model.trainer.re_build_train_loader(
        'dataset_from_seleted_iamge_files')

    # n_batches cycles were used to sample all the data of the training set
    n_batches = int(np.ceil(((train_size-seed_batch) * 1/batch_size))) + 1
    for n in range(n_batches):
        n_train = seed_batch + min((train_size - seed_batch), n * batch_size)
        print('{} data ponints for training in iter{}'.format(n_train, n+1))
        assert n_train == len(seleted_image_files)
        data_sizes.append(n_train)
        ins_seg_model.fit_on_subset(data_loader_from_selected_iamge_files)
        miou = ins_seg_model.test()
        mious.append(miou)
        print('miou：{} in {} iter'.format(miou['miou'], n+1))

        n_sample = min(batch_size, train_size - len(seleted_image_files))
        new_batch = sampler.select_batch(n_sample, already_selected=seleted_image_files)
        seleted_image_files.extend(new_batch)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
        assert len(new_batch) == n_sample

    results['mious'] = mious
    results['data_sizes'] = data_sizes
    results['sampler'] = sampler.sample_name
    print(results)




if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    seg_model = InsSegModel(args=args, project_id='debug', data_dir=data_dir)
    data_loader = seg_model.trainer.data_loader
    randomsampler = RandomSampler('randomsampler', data_loader)
    generate_one_curve(data_loader,
                       sampler=randomsampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.2
                       )