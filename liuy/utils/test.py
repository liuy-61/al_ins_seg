import json
import os
import random

import cv2

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from liuy.utils.local_config import tiny_train, tiny_val, train
from detectron2.structures import BoxMode

def extract_person(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    person_image_id = []
    person_annotations = []
    for annotation in json_data['annotations']:
        if annotation['category_id'] == 1:
            person_annotations.append(annotation)
            person_image_id.append(annotation['image_id'])

    person_image_id = set(person_image_id)

    person_images = []
    for image in json_data['images']:
        if image['id'] in person_image_id:
            person_images.append(image)

    json_data2 = {'annotations': person_annotations, 'images': person_images}

    with open(file_path, 'w') as f:
        json.dump(json_data2, f)

    debug = 1


def get_hw_dicts(file_path, train=True):
    """

    :return: a list[dict], dict : {'file_name': str :'the/path/to/image/2345.jpg',
                                    'height': int,
                                    'width': int,
                                    'image_id': int,
                                    'annotations': list[dict]':
                                                {  'bbox': list[float],
                                                   'bbox_mode': BoxMode.XYWH_ABS,
                                                   'category_id':int,
                                                   'segmentation':list[list[float]] each list[float] is one
                                                   simple polygon in the format of [x1, y1, ...,xn,yn]
                                                   }
    """
    if train is True:
        image_root = '/media/tangyp/Data/coco/train2014'
    else:
        image_root = '/media/tangyp/Data/coco/val2014'

    extract_person(file_path)

    with open(file_path, 'r') as f:
        json_data = json.load(f)
    dict_list = []
    for image in json_data['images']:
        dic = {'file_name': os.path.join(image_root, image['file_name']),
               'height': image['height'],
               'width': image['width'],
               'image_id': image['id']}

        annotations = []
        for annotation in json_data['annotations']:
            if annotation['image_id'] == image['id']:
                d = {'bbox': annotation['bbox'],
                     'bbox_mode': BoxMode.XYWH_ABS,
                     'category_id': annotation['category_id'],
                     'segmentation': annotation['segmentation'],
                     }
                annotations.append(d)
                json_data['annotations'].remove(annotation)
        dic['annotations'] = annotations

        dict_list.append(dic)
    return dict_list


def register_hw_instances(name, file_path, train=True):
    """

    :param name:
    :return:
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: get_hw_dicts(file_path, train))
    MetadataCatalog.get(name).set(thing_classes=["person"])





if __name__ == '__main__':
    # dataset_dicts = get_hw_dicts(file_path=tiny_val, train=False)
    dicts = get_hw_dicts(file_path=train)
    debug = 1
