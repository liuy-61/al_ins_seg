import os
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_cityscapes_instances
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic
from detectron2.engine import default_argument_parser, default_setup
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

import json

# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine2_{task}_train": ("cityscape/leftimage8//leftImg8bit/train", "cityscape/gtFine/train"),
    "cityscapes_fine2_{task}_val": ("cityscape/leftimage8//leftImg8bit/val", "cityscape/gtFine/val"),
    "cityscapes_fine2_{task}_test": ("cityscape/leftimage8//leftImg8bit/test", "cityscape/gtFine/test"),
}


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )

def get_custom_dicts(data_dir):
    if 'train' in data_dir:
        file_path = '/media/tangyp/Data/coco/train2014'
    elif 'val' in data_dir:
        file_path = '/media/tangyp/Data/coco/val2014'
    json_file = data_dir
    coco = COCO(json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    imgs = imgs_anns['images']
    for img in imgs:
        dataset_dict = {}
        # new_img = {'file_name':  '/media/tangyp/Data/coco/train2014' + '/' + img['file_name'], 'height': img['height'], 'width': img['width'],
        #                'image_id': img['id']}
        new_img = {'file_name': os.path.join(file_path, img['file_name']), 'height': img['height'], 'width': img['width'],
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
        debug = 1
    return dataset_dicts


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    data_dir = '/media/tangyp/Data/coco/annotations/instances_train2014.json'
    data_val_dir = '/media/tangyp/Data/coco/annotations/instances_val2014.json'
    DatasetCatalog.register("custom", lambda data_dir=data_dir: get_custom_dicts(data_dir))
    DatasetCatalog.register("custom_val", lambda data_dir=data_val_dir: get_custom_dicts(data_dir))
