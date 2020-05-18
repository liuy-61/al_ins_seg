from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.CoresetSampler import CoreSetSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files, \
    register_coco_instances_from_selected_image_files
from liuy.utils.img_list import save_img_list, read_img_list, get_iter
import time
from liuy.utils.local_cofig import coco_data, debug_data,OUTPUT_DIR
import os



def train_seed(args, project_id, coco_data, resume_or_load, seed_batch):
    """
    check if there is origin (100)image_id list in the OUTPUT_DIR/selected_image_list/project_id  dir
    if not save the origin (100)image_id list
    the file 100 is whole data set image id list
    the file 0 is this iter we randomly select image id list
    """
    dir = OUTPUT_DIR + '/' + 'selected_img_list' + '/' + project_id
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = dir + '/' + str(100)

    if not os.path.exists(file):
        ins_seg_model = CoCoSegModel(
            args=args,
            project_id=project_id,
            coco_data=coco_data,
            resume_or_load=resume_or_load,
        )
        data_loader = ins_seg_model.trainer.data_loader
        image_files_list = []
        index_list = data_loader.dataset._dataset._lst
        for item in index_list:
            image_files_list.append(item['image_id'])
        save_img_list(project_id=project_id, iteration=100, img_id_list=image_files_list)
        print("run the function train_seed again")

    else:
        image_files_list = read_img_list(project_id=project_id, iteration=100)
        whole_train_size = len(image_files_list)
        if seed_batch < 1:
            seed_batch = int(seed_batch * whole_train_size)

        selected_image_files = random.sample(image_files_list, seed_batch)
        print("selected {} images from the {} images ".format(seed_batch, whole_train_size))
        save_img_list(project_id=project_id, iteration=0, img_id_list=selected_image_files)
        print("save the image ids randomly selected this iter 0")

        ins_seg_model = CoCoSegModel(
            args=args,
            project_id=project_id,
            coco_data=coco_data,
            train_size=len(selected_image_files),
            resume_or_load=resume_or_load,
        )
        register_coco_instances_from_selected_image_files(
            name='coco_from_selected_image',
            json_file=coco_data[0]['json_file'],
            image_root=coco_data[0]['image_root'],
            selected_image_files=selected_image_files
        )
        data_loader_from_selected_image_files, _ = ins_seg_model.trainer.re_build_train_loader(
            'coco_from_selected_image')

        ins_seg_model.fit_on_subset(data_loader_from_selected_image_files, iter_num=0)



def train_on_batch(args, project_id, coco_data, resume_or_load, seed_batch, batch_size):
    # get the whole indexes of coco
    image_files_list = read_img_list(project_id=project_id, iteration=100)
    whole_train_size = len(image_files_list)
    if seed_batch < 1:
        seed_batch = int(seed_batch * whole_train_size)
    if batch_size < 1:
        batch_size = int(batch_size * whole_train_size)

    # get the iter_num now by accessing saved indexes eg(if file 0 exist then iter_num now is 1)_
    iter_num = get_iter(project_id=project_id)
    n_batches = int(np.ceil(((whole_train_size - seed_batch) * 1 / batch_size))) + 1

    for n in range(n_batches):
        if n != iter_num:
            continue
        else:
            "" "init seg_model  """
            selected_image_files = read_img_list(project_id=project_id, iteration=iter_num - 1)
            train_size_this_iter = seed_batch + min((whole_train_size - len(selected_image_files)), n * batch_size)
            ins_seg_model = CoCoSegModel(
                args=args,
                project_id=project_id,
                coco_data=coco_data,
                train_size=train_size_this_iter,
                resume_or_load=resume_or_load
            )
            data_loader = ins_seg_model.trainer.data_loader
            mask_feature = ins_seg_model.save_mask_features(json_file=coco_data[0]['json_file'],
                                                            image_root=coco_data[0]['image_root'])
            """ init sampler"""
            # sampler = CoCoRandomSampler('random_sampler', data_loader)
            sampler = CoreSetSampler('coreset_sampler', mask_feature)

            n_sample = min(batch_size, whole_train_size - len(selected_image_files))
            start_time = int(time.time())
            new_batch = sampler.select_batch(n_sample, already_selected=selected_image_files)
            end_time = int(time.time())
            print("select batch using " + str(end_time - start_time) + "s")
            print("selected {} new images in {} iter,{} images used to train".format(n_sample, n, train_size_this_iter))

            selected_image_files.extend(new_batch)
            save_img_list(project_id=project_id, iteration=n, img_id_list=selected_image_files)
            print("save {} images id list ".format(len(selected_image_files)))


            register_coco_instances_from_selected_image_files(
                name='coco_from_selected_image',
                json_file=coco_data[0]['json_file'],
                image_root=coco_data[0]['image_root'],
                selected_image_files=selected_image_files
            )
            data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
                'coco_from_selected_image')

            assert train_size_this_iter == len(selected_image_files)
            ins_seg_model.fit_on_subset(data_loader_from_selected_image_files, iter_num=iter_num)
            print('in {} iter'.format(n))


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    project_id = 'debug_coreset'
    # train_seed(args=args, project_id=project_id, coco_data=debug_data, resume_or_load=True, seed_batch=100)
    train_on_batch(args=args, project_id=project_id, coco_data=debug_data,
                   resume_or_load=True, seed_batch=100, batch_size=100)
