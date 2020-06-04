from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.LossSampler import LossSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files,register_coco_instances_from_selected_image_files
from liuy.utils.local_cofig import coco_data, debug_data

def generate_one_curve(
        whole_image_id_1,
        whole_image_id_2,
        coco_data,
        sampler,
        ins_seg_model_1,
        ins_seg_model_2,
        seed_batch,
        batch_size
):
    # initialize the quantity relationship
    whole_train_size = len(whole_image_id_1)
    if seed_batch < 1:
        seed_batch = int(seed_batch * whole_train_size)
    if batch_size < 1:
        batch_size = int(batch_size * whole_train_size)

    # initally, seed_batch pieces of image were selected randomly
    selected_image_id_1= random.sample(whole_image_id_1, seed_batch)
    selected_image_id_2 = random.sample(whole_image_id_2, seed_batch)
    # register data set and build data loader
    register_coco_instances_from_selected_image_files(name='coco_from_selected_image_1',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_id_1)

    data_loader_1_from_selected_image_files, l = ins_seg_model_1.trainer.re_build_train_loader(
        'coco_from_selected_image_1')

    register_coco_instances_from_selected_image_files(name='coco_from_selected_image_2',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_id_2)

    data_loader_2_from_selected_image_files, l = ins_seg_model_1.trainer.re_build_train_loader(
        'coco_from_selected_image_2')


    n_batches = int(np.ceil(((whole_train_size - seed_batch) * 1 / batch_size))) + 1
    for n in range(n_batches):
        # check the size in this iter
        n_train_size = seed_batch + min((whole_train_size - seed_batch), n * batch_size)
        print('{} data ponints for training in iter{}'.format(n_train_size, n))
        assert n_train_size == len(selected_image_id_1)
        assert len(selected_image_id_1) == len(selected_image_id_2)

        ins_seg_model_1.save_selected_image_id(selected_image_id_1)

        ins_seg_model_1.fit_on_subset(data_loader_1_from_selected_image_files)

        ins_seg_model_2.save_selected_image_id(selected_image_id_2)

        ins_seg_model_2.fit_on_subset(data_loader_2_from_selected_image_files)


        # get the losses for loss_sampler
        print("get losses_1 use ins_seg_model_2")
        losses_1 = ins_seg_model_2.compute_loss(json_file=coco_data[0]['json_file'],
                                                image_root=coco_data[0]['image_root'],)

        print("get losses_2 use ins_seg_model_1")
        losses_2 = ins_seg_model_1.compute_loss(json_file=coco_data[0]['json_file'],
                                                image_root=coco_data[0]['image_root'], )


        n_sample = min(batch_size, whole_train_size - len(selected_image_id_1))

        new_batch_1 = sampler.select_batch(n_sample,
                                           already_selected=selected_image_id_1,
                                           losses=losses_2,
                                           loss_decrease=False)
        selected_image_id_1.extend(new_batch_1)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch_1)))

        new_batch_2 = sampler.select_batch(n_sample,
                                           already_selected=selected_image_id_2,
                                           losses=losses_1,
                                           loss_decrease=False)
        selected_image_id_2.extend(new_batch_2)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch_2)))



        # register dataset and build data loader
        register_coco_instances_from_selected_image_files(name='coco_from_selected_image_1',
                                                          json_file=coco_data[0]['json_file'],
                                                          image_root=coco_data[0]['image_root'],
                                                          selected_image_files=selected_image_id_1)
        data_loader_1_from_selected_image_files, l = ins_seg_model_1.trainer.re_build_train_loader(
            'coco_from_selected_image_1')

        register_coco_instances_from_selected_image_files(name='coco_from_selected_image_2',
                                                          json_file=coco_data[0]['json_file'],
                                                          image_root=coco_data[0]['image_root'],
                                                          selected_image_files=selected_image_id_2)
        data_loader_2_from_selected_image_files, l = ins_seg_model_2.trainer.re_build_train_loader(
            'coco_from_selected_image_2')

        assert len(new_batch_1) == n_sample
        assert len(new_batch_2) == n_sample

        # reset model if necessary
        ins_seg_model_1.reset_model()
        ins_seg_model_2.reset_model()



if __name__ == "__main__":

    data = debug_data
    args = default_argument_parser().parse_args()
    seg_model_1 = CoCoSegModel(args,
                               project_id='co_teaching_model_1',
                               coco_data=data,
                               model_config='Mask_RCNN2',
                               resume_or_load=True)

    seg_model_2 = CoCoSegModel(args,
                               project_id='co_teaching_model_2',
                               coco_data=data,
                               model_config='Mask_RCNN',
                               resume_or_load=True)

    data_loader_1 = seg_model_1.trainer.data_loader

    data_loader_2 = seg_model_2.trainer.data_loader

    whole_image_id_1 = []
    index_list = data_loader_1.dataset._dataset._lst
    for item in index_list:
        whole_image_id_1.append(item['image_id'])

    whole_image_id_2 = []
    index_list = data_loader_2.dataset._dataset._lst
    for item in index_list:
        whole_image_id_2.append(item['image_id'])

    losssampler = LossSampler('loss_sampler')

    generate_one_curve(whole_image_id_1=whole_image_id_1,
                       whole_image_id_2=whole_image_id_2,
                       coco_data=data,
                       sampler=losssampler,
                       ins_seg_model_1=seg_model_1,
                       ins_seg_model_2=seg_model_2,
                       seed_batch=100,
                       batch_size=100)


