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
from alcloud.alcloud.model_updating.interface import BaseDeepModel
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

MODEL_NAME = {'Faster_RCNN': '/home/tangyp/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
              'Mask_RCNN':'/home/tangyp/liuy/detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml'
              }

__all__ = ['Detctron2AlObjDetModel',
           ]
TRAINED_MODEL_DIR = '/media/tangyp/Data/model_file/trained_model'

class Detctron2AlObjDetModel(BaseDeepModel):
    """Faster_RCNN"""

    def __init__(self, args, project_id, model_name=None, num_classes=None, pytorch_model=None):
        self.args = args
        self.project_id = project_id
        self.model_name = model_name
        self.num_classes =num_classes
        self.data_dir = None
        self.lr = None
        # prepare cfg for build model
        self.cfg = setup(args=args, project_id=project_id, model_name=model_name, num_classes=num_classes)
        super(Detctron2AlObjDetModel, self).__init__(project_id)
        self.model, self.device = load_prj_model(project_id=project_id)
        if self.model is None:
            if pytorch_model:
                assert isinstance(
                    pytorch_model, nn.Module), 'pytorch_model must inherit from torch.nn.Module'
                self.model = pytorch_model
                print("get a pre-trained model from parameter for project{}".format(project_id))
            else:
                assert model_name in MODEL_NAME.keys(
                ), 'model_name must be one of {}'.format(MODEL_NAME.keys())
                if not num_classes:
                    raise ValueError(
                        "Deep model of project {} is not initialized, please specify the model name and number of classes.".format(
                            project_id))
                self.model = LiuyTrainer.build_model(self.cfg)
                self.model = self.model.to(self.device)
                print("Initialize a pre-trained model for project{}".format(project_id))
        else:
            print("load project {} model from file".format(project_id))
        print(self.model)

    def fit(self, data_dir, label=None, transform=None,
            batch_size=1, shuffle=False, data_names=None,
            optimize_method='Adam', optimize_param=None,
            loss='CrossEntropyLoss', loss_params=None, num_epochs=10,
            save_model=True, test_label=None, **kwargs):
        self.data_dir = data_dir
        print("Command Line Args:", args)
        self.cfg = setup(args, project_id=self.project_id, model_name=self.model_name, num_classes=self.num_classes,
                         data_dir=data_dir, pre_cfg=self.cfg)
        self.trainer = LiuyTrainer(self.cfg, self.model)
        self.trainer.resume_or_load(resume=args.resume)
        self.trainer.train()
        self.save_model()

    def test(self,data_dir):
        self.data_dir = data_dir
        print("Command Line Args:", args)
        self.cfg = setup(args, project_id=self.project_id, model_name=self.model_name, num_classes=self.num_classes,
                         data_dir=data_dir, pre_cfg=self.cfg)
        self.trainer = LiuyTrainer(self.cfg, self.model)
        self.trainer.test(self.cfg, self.trainer.model)


    def predict_proba(self, data_dir, data_names=None, transform=None, batch_size=1,
                      conf_thres=0.7, nms_thres=0.4,
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
        register_all_cityscapes(data_dir)
        self.cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_test"]
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =conf_thres
        predictor = DefaultPredictor(self.cfg)
        data_loader = LiuyTrainer.build_test_loader(self.cfg, "cityscapes_fine2_instance_seg_test")
        results = []
        for batch in data_loader:
            for item in batch:
                file_name = item['file_name']
                img = cv2.imread(file_name)
                prediction = predictor(img)
                record = {'boxes': prediction['instances'].pred_boxes, 'labels': prediction['instances'].pred_classes, \
                          'scores': prediction['instances'].scores}
                results.append(record)
        return results


    def predict(self, data_dir, data_names=None, transform=None):
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
        proba_result = self.predict_proba(
            data_dir=data_dir, data_names=data_names, transform=transform)
        return proba_result

    # def test(self, data_dir, label, batch_size, **kwargs):
        # self.model_ft.eval()
        # assert isinstance(label, dict)
        # dataloader = create_faster_rcnn_dataloader(data_dir=data_dir, label_dict=label,
        #                                            augment=False, batch_size=batch_size, shuffle=False)
        # with torch.no_grad():
        #     return evaluate(self.model_ft, dataloader, self.device)

    def save_model(self):
        with open(os.path.join(TRAINED_MODEL_DIR, self._proj_id + '_model.pkl'), 'wb') as f:
            pickle.dump(self.trainer.model, f)


def setup(args,project_id,model_name,num_classes=80, lr=0.00025,data_dir=None,pre_cfg=None):
    """
    Create configs and perform basic setups.
    """
    if data_dir is not None:
        register_all_cityscapes(data_dir)
        pre_cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_train"]
        pre_cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_val"]
        return pre_cfg
    cfg = get_cfg()
    config_file = MODEL_NAME[model_name]
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = lr
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.MASK_ON = True
    cfg.OUTPUT_DIR = os.path.join('/media/tangyp/Data/model_file/OUTPUT_DIR','project'+project_id)
    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    model = Detctron2AlObjDetModel(args=args, project_id='4_cityscapes_miou_metric', model_name='Mask_RCNN', num_classes=1)
    # model.fit(data_dir)
    model.test(data_dir)
    debug = 1
