from detectron2.engine import default_argument_parser
# from liuy.implementation.CitySegModel import CitySegModel
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import  CoCoRandomSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files,register_coco_instances_from_selected_image_files
from liuy.utils.img_list import save_img_list,read_img_list,get_iter

def train_on_seed(args, project_id, coco_data, resume_or_load,seed_batch):
    "" "init seg_model  """
    ins_seg_model = CoCoSegModel(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=resume_or_load)
    # get all the image files from the data_loader
    data_loader = ins_seg_model.trainer.data_loader
    image_files_list = []
    list = data_loader.dataset._dataset._lst
    for item in list:
        image_files_list.append(item['image_id'])

    # The size of the entire training set
    train_size = len(image_files_list)
    # transform seed_batch and batch_size from float which indicate percentage of entire training set to int
    seed_batch = int(seed_batch * train_size)

    # We recorded the results of the model training and testing after each data sampling
    # initally, seed_batch pieces of image were selected randomly
    selected_image_files = random.sample(image_files_list, seed_batch)
    save_img_list(project_id=project_id, iteration=0, img_id_list=selected_image_files)
    register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_files)
    data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image')

    ins_seg_model.fit_on_subset(data_loader_from_selected_image_files,iteration=0)
    miou = ins_seg_model.test()
    print('after training  on seed data the miou is: ', miou)

def train_on_batch(args, project_id, coco_data, resume_or_load,seed_batch,batch_size):
    "" "init seg_model  """
    ins_seg_model = CoCoSegModel(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=resume_or_load)
    # get all the image files from the data_loader
    data_loader = ins_seg_model.trainer.data_loader
    sampler = CoCoRandomSampler('randomsampler', data_loader)
    image_files_list = []
    list = data_loader.dataset._dataset._lst
    for item in list:
        image_files_list.append(item['image_id'])

    # The size of the entire training set
    train_size = len(image_files_list)
    # transform seed_batch and batch_size from float which indicate percentage of entire training set to int
    seed_batch = int(seed_batch * train_size)
    batch_size = int(batch_size * train_size)

    n_batches = int(np.ceil(((train_size - seed_batch) * 1 / batch_size))) + 1
    for n in range(n_batches):
        iter = get_iter(project_id)
        if n != iter:
            continue
        else:
            n_train = seed_batch + min((train_size - seed_batch), n * batch_size)
            print('{} data ponints for training in iter{}'.format(n_train, n))
            # get the last saved selected_image_files
            selected_image_files = read_img_list(project_id=project_id, iteration=n-1)
            # use the sampler to sample data
            n_sample = min(batch_size, train_size - len(selected_image_files))
            new_batch = sampler.select_batch(n_sample, already_selected=selected_image_files)
            selected_image_files.extend(new_batch)
            # save the selected_image_files for this iter
            save_img_list(project_id=project_id, iteration=n, img_id_list=selected_image_files)

            register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                              json_file=coco_data[0]['json_file'],
                                                              image_root=coco_data[0]['image_root'],
                                                              selected_image_files=selected_image_files)
            data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
                'coco_from_selected_image')

            assert n_train == len(selected_image_files)
            ins_seg_model.fit_on_subset(data_loader_from_selected_image_files,iteration=n)
            miou = ins_seg_model.test()
            print('miou：{} in {} iter'.format(miou['miou'],  n))
            break


if __name__ == "__main__":
    coco_data = [{'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                  'image_root': '/media/tangyp/Data/coco/train2014'
                  },
                 {
                     # 'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                     'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',
                     'image_root': '/media/tangyp/Data/coco/val2014'
                 }]
    args = default_argument_parser().parse_args()
    project_id = 'Base'
    # randomsampler = CoCoRandomSampler('randomsampler', data_loader)
    train_on_seed(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=True, seed_batch=0.1)
    # train_on_batch(args=args, project_id=project_id, coco_data=coco_data, resume_or_load=False, seed_batch=0.1, batch_size=0.05)
