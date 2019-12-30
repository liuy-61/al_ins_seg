import os
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_cityscapes_instances
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic, cityscapes_files_to_dict
from detectron2.engine import default_argument_parser, default_setup
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import glob
import json
import logging
import multiprocessing as mp
import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import get_world_size
from fvcore.common.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

import json

# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine2_{task}_train": ("cityscape/leftImg8bit/train", "cityscape/gtFine/train"),
    "cityscapes_fine2_{task}_val": ("cityscape/leftImg8bit/val", "cityscape/gtFine/val"),
    "cityscapes_fine2_{task}_test": ("cityscape/leftImg8bit/test", "cityscape/gtFine/test"),
    "cityscapes_fine2_{task}_sub_train": ("cityscape/leftImg8bit/sub_train", "cityscape/gtFine/sub_train"),
}


def register_all_cityscapes(root):
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

def register_a_cityscapes(image_dir, gt_dir, dataset_name):
    meta = _get_builtin_metadata("cityscapes")
    DatasetCatalog.register(
        dataset_name,
        lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
            x, y, from_json=True, to_polygons=True
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
    )

def register_a_cityscapes_from_selected_image_files(image_dir, gt_dir, selected_image_files ,dataset_name):
    meta = _get_builtin_metadata("cityscapes")
    DatasetCatalog.register(
        dataset_name,
        lambda x=image_dir, y=gt_dir, z=selected_image_files: load_cityscapes_instances_from_selected_image_files(
            x, y, z, from_json=True, to_polygons=True
        ),
    )
    MetadataCatalog.get(dataset_name).set(
        image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
    )

def load_cityscapes_instances_from_selected_image_files(image_dir, gt_dir, selected_image_files,from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    files = []
    for image_file in selected_image_files:
        suffix = "leftImg8bit.png"
        assert image_file.endswith(suffix)
        prefix = image_dir
        instance_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_instanceIds.png"
        assert os.path.isfile(instance_file), instance_file

        label_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_labelIds.png"
        assert os.path.isfile(label_file), label_file

        json_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_polygons.json"
        files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)

    logger = logging.getLogger(__name__)
    logger.info("Preprocessing cityscapes annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(cityscapes_files_to_dict, from_json=from_json, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map cityscape ids to contiguous ids
    from cityscapesScripts.cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret
