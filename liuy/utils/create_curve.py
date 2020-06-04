import os
import pickle

from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.utils.reg_dataset import register_coco_instances_from_selected_image_files
import random
import numpy as np
from liuy.implementation import RandomSampler
from liuy.utils.local_cofig import OUTPUT_DIR

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

    # initialize the container
    results = {}
    data_sizes = []
    mious = []

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
        data_sizes.append(n_train_size)

        ins_seg_model.save_selected_image_id(selected_image_id)

        ins_seg_model.fit_on_subset(data_loader_from_selected_image_files)
        miou = ins_seg_model.test()
        mious.append(miou)
        print('miou：{} in {} iter'.format(miou['miou'], n))

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

    results['mious'] = mious
    results['data_sizes'] = data_sizes
    print(results)

def generate_base_model(
        whole_image_id,
        coco_data,
        ins_seg_model,
        seed_batch,
        batch_size
):
    """
    generate base models, separately use 20% data 30% data 40% data 40% data 50% data ~~~~100% data
    the data is randomly selected
    and the eavl results save as baseline
    """
    # initialize quantity relationship
    whole_train_size = len(whole_image_id)
    if seed_batch < 1:
        seed_batch = int(seed_batch * whole_train_size)
    if batch_size < 1:
        batch_size = int(batch_size * whole_train_size)

    # initialize random sampler
    sampler = CoCoRandomSampler(sampler_name='random', whole_image_id=whole_image_id)

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

        n_sample = min(batch_size, whole_train_size - len(selected_image_id))
        new_batch = sampler.select_batch(n_sample, already_selected=selected_image_id)

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


def load_base_model(project_id,
                    serial_num):
    """

    :param project_id:
    :param serial_num:
    :return: generalized rcnn
    """
    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id, str(serial_num))
    detail_file = os.path.join(detail_output_dir, project_id + '_model.pkl')

    if os.path.exists(detail_file):
        with open(detail_file, 'rb') as f:
            return pickle.load(f)


def read_selected_image_id(project_id,
                    serial_num):
    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id, str(serial_num))
    detail_file = os.path.join(detail_output_dir, 'selected_image_id.pkl')

    if os.path.exists(detail_file):
        with open(detail_file, 'rb') as f:
            return pickle.load(f)