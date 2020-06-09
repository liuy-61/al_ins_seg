from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_coco_instances_from_selected_image_files
from liuy.utils.local_config import coco_data, debug_data
import copy

def generate_one_curve(
        whole_image_id,
        coco_data,
        sampler,
        ins_seg_model,
        seed_batch,
        batch_size
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

    # initally, seed_batch pieces of image were selected randomly
    selected_image_id = random.sample(whole_image_id, seed_batch)
    # register data set and build data loader
    register_coco_instances_from_selected_image_files(
        name='coco_from_selected_image',
        json_file=coco_data[0]['json_file'],
        image_root=coco_data[0]['image_root'],
        selected_image_files=selected_image_id
    )
    data_loader_from_selected_image_files, _ = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image')

    n_batches = int(np.ceil(((whole_train_size - seed_batch) * 1 / batch_size))) + 1
    for n in range(n_batches):
        # check the size in this iter
        n_train_size = seed_batch + min((whole_train_size - seed_batch), n * batch_size)
        assert n_train_size == len(selected_image_id)
        print('{} data ponints for training in iter{}'.format(n_train_size, n))

        # start training and test
        ins_seg_model.save_selected_image_id(selected_image_id)
        ins_seg_model.fit_on_subset(data_loader_from_selected_image_files)

        # select new batch
        n_sample = min(batch_size, whole_train_size - len(selected_image_id))
        new_batch = sampler.select_batch(n_sample=n_sample, already_selected=copy.deepcopy(selected_image_id))

        selected_image_id.extend(new_batch)
        assert len(new_batch) == n_sample
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))

        # register dataset and build data loader
        register_coco_instances_from_selected_image_files(
            name='coco_from_selected_image',
            json_file=coco_data[0]['json_file'],
            image_root=coco_data[0]['image_root'],
            selected_image_files=selected_image_id
        )
        data_loader_from_selected_image_files, _ = ins_seg_model.trainer.re_build_train_loader(
            'coco_from_selected_image')

        # reset model
        print("--reset model")
        ins_seg_model.reset_model()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    project_id = "random"
    seg_model = CoCoSegModel(args, project_id=project_id, coco_data=debug_data, resume_or_load=True)
    data_loader = seg_model.trainer.data_loader
    whole_image_id = []
    index_list = data_loader.dataset._dataset._lst
    for item in index_list:
        whole_image_id.append(item['image_id'])

    randomsampler = CoCoRandomSampler("random_sampler", whole_image_id=whole_image_id)
    generate_one_curve(
        coco_data=copy.deepcopy(debug_data),
        whole_image_id=copy.deepcopy(whole_image_id),
        sampler=randomsampler,
        ins_seg_model=seg_model,
        batch_size=100,
        seed_batch=100,
    )
