import os
import time

from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.LossSampler import LossSampler
import numpy as np
import random

from liuy.utils.K_means import read_image2class
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files, \
    register_coco_instances_from_selected_image_files
from liuy.utils.local_config import coco_data, debug_data, VAE_feature_path
from liuy.utils.K_means import k_means


def generate_one_curve(
        whole_image_id,
        coco_data,
        sampler,
        ins_seg_model,
        seed_batch,
        batch_size,
        image2class,
):
    """
    :return:
    """
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

        """ get the mask feature use the trained model 
            and use the mask feature to cluster: KNN
        """

        # get the losses for loss_sampler
        losses = ins_seg_model.compute_loss(json_file=coco_data[0]['json_file'],
                                            image_root=coco_data[0]['image_root'], )

        n_sample = min(batch_size, whole_train_size - len(selected_image_id))

        new_batch = sampler.slect_batch_from_groups(n_sample=n_sample,
                                                    already_selected=selected_image_id,
                                                    losses=losses,
                                                    loss_decrease=False,
                                                    image2class=image2class,
                                                    )

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


if __name__ == "__main__":
    coco_data = debug_data

    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='self_paced_with_diversity', coco_data=coco_data, resume_or_load=True)
    data_loader = seg_model.trainer.data_loader
    whole_image_id = [item['image_id'] for item in data_loader.dataset._dataset._lst]

    # waiting for VAE feature to be generated
    while True:
        if not os.path.exists(VAE_feature_path):
            print('waiting for  VAE feature')
            time.sleep(15)
        else:
            break
    print('the VAE feature has been generated')

    # use the VAE feature to cluster
    k_means(feature_path=VAE_feature_path, k=50)
    image2class = read_image2class(k=50)

    losssampler = LossSampler('loss_sampler')
    generate_one_curve(coco_data=coco_data,
                       whole_image_id=whole_image_id,
                       sampler=losssampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.1,
                       image2class=image2class
                       )
