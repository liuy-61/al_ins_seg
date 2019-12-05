
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
import json


def get_custom_dicts(data_dir):
    json_file = data_dir
    coco = COCO(json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    imgs = imgs_anns['images']
    for img in imgs:
        dataset_dict = {}
        new_img = {'file_name':  '/media/tangyp/Data/coco/train2014' + '/' + img['file_name'], 'height': img['height'], 'width': img['width'],
                       'image_id': img['id']}
        annId = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ids=annId)
        annotation = {}
        annotation['annotations'] = []
        for ann in anns:
            new_ann = {'iscrowd': ann['iscrowd'], 'bbox': ann['bbox'], 'category_id': ann['category_id'],
                            'segmentation': ann['segmentation'], 'bbox_mode': BoxMode(1)}
            annotation['annotations'].append(new_ann)
            dataset_dict.update(new_img)
            dataset_dict.update(annotation)
            dataset_dicts.append(dataset_dict)
    return dataset_dicts

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    data_dir = '/media/tangyp/Data/coco/annotations/instances_train2014.json'
    data_val_dir = '/media/tangyp/Data/coco/annotations/instances_val2014.json'
    DatasetCatalog.register("custom", lambda data_dir=data_dir: get_custom_dicts(data_dir))
    DatasetCatalog.register("custom_val", lambda data_dir=data_val_dir: get_custom_dicts(data_dir))
