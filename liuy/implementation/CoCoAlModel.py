from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.LossSampler import LossSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files,register_coco_instances_from_selected_image_files


def generate_one_curve(
        coco_data,
        data_loader,
        sampler,
        ins_seg_model,
        seed_batch,
        batch_size
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
        image_files_list.append(item['image_id'])

    # The size of the entire training set
    train_size = len(image_files_list)
    # transform seed_batch and batch_size from float which indicate percentage of entire training set to int
    seed_batch = int(seed_batch * train_size)
    batch_size = int(batch_size * train_size)

    # We recorded the results of the model training and testing after each data sampling
    results = {}
    data_sizes = []
    mious = []

    # initally, seed_batch pieces of image were selected randomly
    selected_image_files = random.sample(image_files_list, seed_batch)

    register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_files)
    data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image')
    # data_loader_iter = iter(data_loader_from_selected_image_files)
    # data = next(data_loader_iter)
    # n_batches cycles were used to sample all the data of the training set
    n_batches = int(np.ceil(((train_size - seed_batch) * 1 / batch_size))) + 1
    for n in range(n_batches):
        n_train = seed_batch + min((train_size - seed_batch), n * batch_size)
        print('{} data ponints for training in iter{}'.format(n_train, n))
        assert n_train == len(selected_image_files)
        data_sizes.append(n_train)
        ins_seg_model.fit_on_subset(data_loader_from_selected_image_files,n)
        miou = ins_seg_model.test()
        mious.append(miou)
        print('miou：{} in {} iter'.format(miou['miou'], n))

        # get the losses for loss_sampler
        losses = ins_seg_model.compute_loss(json_file=coco_data[0]['json_file'],
                                            image_root=coco_data[0]['image_root'],)


        n_sample = min(batch_size, train_size - len(selected_image_files))
        new_batch = sampler.select_batch(n_sample, already_selected=selected_image_files, losses=losses)
        selected_image_files.extend(new_batch)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
        register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                          json_file=coco_data[0]['json_file'],
                                                          image_root=coco_data[0]['image_root'],
                                                          selected_image_files=selected_image_files)
        data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
            'coco_from_selected_image')
        assert len(new_batch) == n_sample

    results['mious'] = mious
    results['data_sizes'] = data_sizes
    results['sampler'] = sampler.sample_name
    print(results)



def reset_seg_model(seg_model,coco_data):
    args = seg_model.args
    project_id = seg_model.project_id
    resume_or_load = seg_model.resume_or_load
    del seg_model
    new_seg_model = CoCoSegModel(args, project_id, coco_data, resume_or_load)
    return new_seg_model


if __name__ == "__main__":
    coco_data = [{#'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                  'json_file': '/media/tangyp/Data/coco/annotations/sub_train2014.json', # a subset of train set
                  'image_root': '/media/tangyp/Data/coco/train2014'
                  },
                 {
                     # 'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                     'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',# a subset of val set
                     'image_root': '/media/tangyp/Data/coco/val2014'
                 }]
    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='test', coco_data=coco_data, resume_or_load=False)
    data_loader = seg_model.trainer.data_loader
    losssampler = LossSampler('loss_sampler', data_loader)
    generate_one_curve(coco_data=coco_data,
                       data_loader=data_loader,
                       sampler=losssampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.4,
                       )
