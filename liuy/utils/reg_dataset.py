from detectron2.data.datasets import load_cityscapes_instances
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic, cityscapes_files_to_dict
from pycocotools.coco import COCO
import functools
import multiprocessing as mp
from detectron2.utils import logger
from detectron2.utils.comm import get_world_size
import io
import logging
import contextlib
import os
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

import json

# ==== Predefined splits for raw cityscapes images ===========

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    ]

_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine2_{task}_train": ("cityscape/leftImg8bit/train", "cityscape/gtFine/train"),
    "cityscapes_fine2_{task}_val": ("cityscape/leftImg8bit/val", "cityscape/gtFine/val"),
    "cityscapes_fine2_{task}_test": ("cityscape/leftImg8bit/test", "cityscape/gtFine/test"),
    "cityscapes_fine2_{task}_sub_train": ("cityscape/leftImg8bit/sub_train", "cityscape/gtFine/sub_train"),
}

def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name == "coco_person":
        return _get_coco_instances_meta()
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle", "license plate",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
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


def get_coco_dicts_from_selected_image_files(json_file, image_root, selected_image_files,
                                             dataset_name=None, extra_annotation_keys=None):

    dataset_dicts = get_coco_person_dicts(json_file=json_file,
                                          image_root=image_root,
                                          dataset_name=dataset_name,
                                          extra_annotation_keys=extra_annotation_keys)

    dataset_dicts = [item for item in dataset_dicts if item['image_id'] in selected_image_files]
    return dataset_dicts


def register_coco_instances_from_selected_image_files(name, json_file,  image_root, selected_image_files):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: get_coco_dicts_from_selected_image_files(json_file, image_root, selected_image_files, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    metadata = _get_builtin_metadata('coco_person')
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def get_coco_person_dicts(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
        get a list of dicts, the dict only contain person class img and person ann

        Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str): the directory where the images in this json file exists.
            dataset_name (str): the name of the dataset (e.g., coco_2017_train).
                If provided, this function will also put "thing_classes" into
                the metadata associated with this dataset.
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
                "category_id", "segmentation"). The values for these keys will be returned as-is.
                For example, the densepose annotations are loaded in this way.

        Returns:
            list[dict]: a list of dicts in Detectron2 standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )

        Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
        """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # cat_ids = sorted(coco_api.getCatIds())
        """
        fix the category as person 
        """
        cat_ids = coco_api.getCatIds('person')
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    # img_ids = sorted(list(coco_api.imgs.keys()))
    """ fix the img_ids and sort it
    """
    img_ids = coco_api.getImgIds(catIds=cat_ids)
    img_ids = sorted(img_ids)

    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            if anno['category_id'] == 1:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0
                obj = {key: anno[key] for key in ann_keys if key in anno}

                segm = anno.get("segmentation", None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                keypts = anno.get("keypoints", None)
                if keypts:  # list[int]
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            keypts[idx] = v + 0.5
                    obj["keypoints"] = keypts

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warn(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    debug = 1
    return dataset_dicts
    # dataset_person_dicts = []
    # for dataset_dict in dataset_dicts:
    #     if dataset_dict['image_id'] in person_img_ids:
    #         dataset_person_dicts.append(dataset_dict)
    #
    # assert len(person_img_ids) == len(dataset_person_dicts)
    # debug = 1
    # return dataset_person_dicts


def register_coco_instances(name, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: get_coco_person_dicts(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    metadata = _get_builtin_metadata('coco_person')
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def get_hw_dicts(image_id=None):
    """
    image_id: list[int], if given image_id, the returned dict_list only contain corresponding dict.
    :return: a list[dict], dict : {'file_name': str :'the/path/to/image/2345.jpg',
                                    'height': int,
                                    'width': int,
                                    'image_id': int,
                                    'annotations': list[dict]':
                                                {  'bbox': list[float],
                                                   'bbox_mode': int,
                                                   'category_id':int,
                                                   'segmentation':list[list[float]] each list[float] is one
                                                   simple polygon in the format of [x1, y1, ...,xn,yn]
                                                   }
    """
    dict_list = []
    if image_id is not None:
        dict_list = [dic for dic in dict_list if dic['image_id'] in image_id]
    return dict_list


def register_hw_instances(name, image_id=None):
    """

    :param image_id: see function get_hw_dicts
    :param name:
    :return:
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: get_hw_dicts(image_id))
    MetadataCatalog.get(name).set(thing_classes=["person"])



