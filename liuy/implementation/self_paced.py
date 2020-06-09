from detectron2.engine import default_argument_parser
from liuy.implementation.CoCoSegModel import CoCoSegModel
from liuy.implementation.RandomSampler import CoCoRandomSampler
from liuy.implementation.LossSampler import LossSampler
import numpy as np
import random
from liuy.utils.reg_dataset import register_a_cityscapes_from_selected_image_files,register_coco_instances_from_selected_image_files
from liuy.utils.local_config import coco_data, debug_data

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
    register_coco_instances_from_selected_image_files(name='coco_from_selected_image_id',
                                                      json_file=coco_data[0]['json_file'],
                                                      image_root=coco_data[0]['image_root'],
                                                      selected_image_files=selected_image_id)
    data_loader_from_selected_image_files, l = ins_seg_model.trainer.re_build_train_loader(
        'coco_from_selected_image_id')

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

        new_batch = sampler.select_batch(n_sample,
                                         already_selected=selected_image_id,
                                         losses=losses,
                                         loss_decrease=False)
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



if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='debug', coco_data=debug_data, resume_or_load=True)
    data_loader = seg_model.trainer.data_loader
    whole_image_id = []
    index_list = data_loader.dataset._dataset._lst
    for item in index_list:
        whole_image_id.append(item['image_id'])

    losssampler = LossSampler('loss_sampler')
    generate_one_curve(coco_data=debug_data,
                       whole_image_id=whole_image_id,
                       sampler=losssampler,
                       ins_seg_model=seg_model,
                       batch_size=100,
                       seed_batch=100,
                       )
