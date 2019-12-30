import cv2
import torch
import torch.nn as nn
import logging
import os
from detectron2.utils.visualizer import Visualizer
import dill as pickle
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup, DefaultPredictor
from detectron2.config.config import get_cfg
from liuy.interface import BaseInsSegModel
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser
from detectron2.evaluation import verify_results, SemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, \
    CityscapesEvaluator, PascalVOCDetectionEvaluator, LVISEvaluator, DatasetEvaluators
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils import comm
from liuy.reg_dataset1 import get_custom_dicts,register_all_cityscapes
from detectron2.engine.defaults import DefaultTrainer
from alcloud.alcloud.utils.data_manipulate import create_img_dataloader, create_faster_rcnn_dataloader
from alcloud.alcloud.utils.detection.engine import evaluate
from alcloud.alcloud.utils.torch_utils import load_prj_model
from liuy.LiuyTrainer import  LiuyTrainer
from liuy.Liuy_loss import LiuyTensorboardXWriter
from liuy.reg_dataset1 import register_a_cityscapes
from liuy.ComputeLoss import LiuyComputeLoss
# the  config file of the model
MODEL_NAME = {'Faster_RCNN': '/home/tangyp/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
              'Mask_RCNN':'/home/tangyp/liuy/detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml'
              }

__all__ = ['InsSegModel',
          ]
# the dir where to save the model
TRAINED_MODEL_DIR = '/media/tangyp/Data/model_file/trained_model'

class InsSegModel(BaseInsSegModel):
    """Mask_RCNN"""

    def __init__(self, args, project_id, data_dir):
        self.args = args
        super(InsSegModel, self).__init__(project_id, data_dir)
        # tow ways to get model
        # 1：load the model which has been trained
        # 2：use the function：LiuyTrainer.build_model(self.cfg)
        self.model, self.device = load_prj_model(project_id=project_id)
        self.cfg = setup(args=args, project_id=project_id, data_dir=data_dir)
        if self.model is None:
                self.model = LiuyTrainer.build_model(self.cfg)
                self.model = self.model.to(self.device)
                print("Initialize a pre-trained model for project{}".format(project_id))
        else:
            print("load project {} model from file".format(project_id))
        self.trainer = LiuyTrainer(self.cfg, self.model)

    def fit(self):
        self.cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_train"]
        self.trainer = LiuyTrainer(self.cfg, self.model)
        self.trainer.resume_or_load(resume=args.resume)
        self.trainer.train()
        self.save_model()

    def fit_on_subset(self, **kwargs):
        self.cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_sub_train"]
        self.trainer = LiuyTrainer(self.cfg, self.model)
        self.trainer.resume_or_load(resume=args.resume)
        self.trainer.train()
        self.save_model()

    def test(self):
        self.cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_val"]
        self.trainer = LiuyTrainer(self.cfg, self.model)
        self.trainer.test(self.cfg, self.trainer.model)


    def compute_loss(self, image_dir, gt_dir):
        register_a_cityscapes(image_dir, gt_dir, 'dataset_name')
        self.cfg.DATASETS.TRAIN = ["dataset_name"]
        computer = LiuyComputeLoss(self.cfg, self.model)
        return computer.compute()


    def predict_proba(self, image_dir, gt_dir, conf_thres=0.7, nms_thres=0.4,
                      verbose=True, **kwargs):
        """
                   During inference, the model requires only the input tensors, and returns the post-processed
                   predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
                   follows:
                       - boxes (Tensor[N, 4]): the predicted boxes in [x0, y0, x1, y1] format, with values between
                         0 and H and 0 and W
                       - labels (Tensor[N]): the predicted labels for each image
                       - scores (Tensor[N]): the scores or each prediction
        """
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        register_a_cityscapes(image_dir, gt_dir, 'dataset_name')
        predictor = DefaultPredictor(self.cfg)
        data_loader = LiuyTrainer.build_test_loader(self.cfg, "dataset_name")
        results = []
        for batch in data_loader:
            for item in batch:
                file_name = item['file_name']
                img = cv2.imread(file_name)
                prediction = predictor(img)
                record = {'file_name': file_name, 'boxes': prediction['instances'].pred_boxes, 'labels': prediction['instances'].pred_classes, \
                          'scores': prediction['instances'].scores}
                results.append(record)
        return results




    def predict(self, image_dir, gt_dir):
        '''predict

        :param data_dir: str
            The path to the data folder.

        :param data_names: list, optional (default=None)
            The data names. If not specified, it will all the files in the
            data_dir.

        :param transform: torchvision.transforms.Compose, optional (default=None)
            Transforms object that will be applied to the image data.

        :return: pred: 1D array
            The prediction result. Shape [n_samples]
        '''
        proba_result = self.predict_proba(image_dir, gt_dir)
        return proba_result


    def save_model(self):
        with open(os.path.join(TRAINED_MODEL_DIR, self._proj_id + '_model.pkl'), 'wb') as f:
            pickle.dump(self.trainer.model, f)




def setup(args,project_id,data_dir=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    config_file = MODEL_NAME['Mask_RCNN']
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.MASK_ON = True
    cfg.OUTPUT_DIR = os.path.join('/media/tangyp/Data/model_file/OUTPUT_DIR','project'+project_id)
    register_all_cityscapes(data_dir)
    cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_val"]
    cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_train"]
    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    model = InsSegModel(args=args, project_id='baseline', data_dir=data_dir)
    model.fit()
    losses = model.compute_loss(image_dir=image_dir,gt_dir=gt_dir)
    # probability = model.predict_probability(image_dir, gt_dir)
    model.test()
    # debug = 1
