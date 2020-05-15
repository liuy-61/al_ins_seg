from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files, \
    register_coco_instances_from_selected_image_files
from liuy.utils.img_list import save_img_list, read_img_list, get_iter
import time
from liuy.utils.seed_torch import seed_torch

def train_on_seed(args, project_id, coco_data, resume_or_load, seed_batch):
    "" "init seg_model  """
    # ins_seg_model = CoCoSegModel(
    #     args=args,
    #     project_id=project_id,
    #     coco_data=coco_data,
    #     resume_or_load=resume_or_load,
    # )
    # data_loader = ins_seg_model.trainer.data_loader
    # image_files_list = []
    # index_list = data_loader.dataset._dataset._lst
    # for item in index_list:
    #     image_files_list.append(item['image_id'])
    # save_img_list(project_id=project_id, iteration=100, img_id_list=image_files_list)

    image_files_list = read_img_list(project_id=project_id, iteration=100)
    train_size = len(image_files_list)
    if seed_batch < 1:
        seed_batch = int(seed_batch * train_size)

    selected_image_files = random.sample(image_files_list, seed_batch)
    save_img_list(project_id=project_id, iteration=0, img_id_list=selected_image_files)

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
    data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image')

    ins_seg_model.fit_on_subset(data_loader_from_selected_image_files)
    miou = ins_seg_model.test()
    print('after training  on seed data the miou is: ', miou)


def train_on_batch(args, project_id, coco_data, resume_or_load, seed_batch, batch_size):
    # get the whole indexes of coco
    image_files_list = read_img_list(project_id=project_id, iteration=100)
    train_size = len(image_files_list)
    if seed_batch < 1:
        seed_batch = int(seed_batch * train_size)
    if batch_size < 1:
        batch_size = int(batch_size * train_size)

    # get the iter_num now by accessing saved indexes
    iter_num = get_iter(project_id=project_id)
    n_batches = int(np.ceil(((train_size - seed_batch) * 1 / batch_size))) + 1

    for n in range(n_batches):
        if n != iter_num:
            continue
        else:
            "" "init seg_model  """
            n_train = seed_batch + min((train_size - seed_batch), n * batch_size)
            ins_seg_model = CoCoSegModel(
                args=args,
                project_id=project_id,
                coco_data=coco_data,
                train_size=n_train,
                resume_or_load=resume_or_load
            )
            data_loader = ins_seg_model.trainer.data_loader
            sampler = CoCoRandomSampler('random_sampler', data_loader)
            print('{} data ponints for training in iter{}'.format(n_train, n))

            selected_image_files = read_img_list(project_id=project_id, iteration=iter_num - 1)
            n_sample = min(batch_size, train_size - len(selected_image_files))
            start_time = int(time.time())
            new_batch = sampler.select_batch(n_sample, already_selected=selected_image_files)
            end_time = int(time.time())
            print("select batch using " + str(end_time - start_time) + "s")

            selected_image_files.extend(new_batch)
            save_img_list(project_id=project_id, iteration=n, img_id_list=selected_image_files)
            register_coco_instances_from_selected_image_files(
                name='coco_from_selected_image',
                json_file=coco_data[0]['json_file'],
                image_root=coco_data[0]['image_root'],
                selected_image_files=selected_image_files
            )
            data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
                'coco_from_selected_image')

            assert n_train == len(selected_image_files)
            ins_seg_model.fit_on_subset(data_loader_from_selected_image_files, iter_num=iter_num)
            miou = ins_seg_model.test()
            print('miou：{} in {} iter'.format(miou['miou'], n))


if __name__ == "__main__":
    seed_torch()
    coco_data = [{'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                  'image_root': '/media/tangyp/Data/coco/train2014'
                  },
                 {
                     'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                     # 'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',
                     'image_root': '/media/tangyp/Data/coco/val2014'
                 }]
    args = default_argument_parser().parse_args()
    project_id = 'random_100'
    # randomsampler = CoCoRandomSampler('randomsampler', data_loader)
    train_on_seed(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=True, seed_batch=100)
    # train_on_batch(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=True, seed_batch=0.2, batch_size=0.1)
