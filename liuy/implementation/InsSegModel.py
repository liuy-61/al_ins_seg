import cv2
import os
import dill as pickle
from detectron2.engine import default_setup, DefaultPredictor
from detectron2.config.config import get_cfg
from liuy.Interface.BaseInsSegModel import BaseInsSegModel
from detectron2.engine import default_argument_parser
from liuy.utils.reg_dataset import register_a_cityscapes,register_all_cityscapes
from liuy.utils.torch_utils import load_prj_model
from liuy.utils.torch_utils import OUTPUT_DIR
from liuy.utils.LiuyTrainer import LiuyTrainer
from liuy.utils.ComputeLoss import LiuyComputeLoss
# the  config file of the model
MODEL_NAME = {'Faster_RCNN': '/home/tangyp/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
              'Mask_RCNN':'/home/tangyp/liuy/detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml'
              }

__all__ = ['InsSegModel',
          ]
# the dir where to save the model
# TRAINED_MODEL_DIR = '/media/tangyp/Data/model_file/trained_model'
# OUTPUT_DIR = '/media/tangyp/Data/model_file/OUTPUT_DIR'
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
        self.trainer.resume_or_load(resume=self.args.resume)
        self.trainer.train()
        self.save_model()

    def fit_on_subset(self, data_loader):
        self.trainer.resume_or_load(resume=self.args.resume)
        self.trainer.data_loader = data_loader
        self.trainer._data_loader_iter = iter(data_loader)
        self.trainer.train()
        self.save_model()

    def test(self):
        """
        :return: test result on val
        """
        self.cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_val"]
        self.cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_train"]
        self.trainer = LiuyTrainer(self.cfg, self.model)
        miou = self.trainer.test(self.cfg, self.trainer.model)
        return miou


    def compute_loss(self, image_dir, gt_dir):
        """
        :param image_dir:
        :param gt_dir:
        :return: list of dict, a dict includes file_name（as a key to the data) and losses
        """
        register_a_cityscapes(image_dir, gt_dir, 'dataset_name')
        self.cfg.DATASETS.TRAIN = ["dataset_name"]
        computer = LiuyComputeLoss(self.cfg, self.model)
        return computer.compute()


    def predict_proba(self, image_dir, gt_dir, conf_thres=0.7, nms_thres=0.4,
                      verbose=True, **kwargs):
        """
                :return: list of dict, a dict includes file_name（as a key to the data) and predictions
        """
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        register_a_cityscapes(image_dir, gt_dir, 'dataset_name')
        predictor = DefaultPredictor(self.cfg)
        data_loader, data_len = LiuyTrainer.build_test_loader(self.cfg, "dataset_name")
        results = []
        for batch in data_loader:
            for item in batch:
                file_name = item['file_name']
                img = cv2.imread(file_name)
                prediction = predictor(img)
                record = {'file_name': file_name, 'boxes': prediction['instances'].pred_boxes, 'labels': prediction['instances'].pred_classes, \
                          'scores': prediction['instances'].scores, 'masks':prediction['instances'].pred_masks}
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
        with open(os.path.join(OUTPUT_DIR, self._proj_id + '_model.pkl'), 'wb') as f:
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
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'project'+project_id)
    register_all_cityscapes(data_dir)
    cfg.DATASETS.TEST = ["cityscapes_fine2_instance_seg_val"]
    cfg.DATASETS.TRAIN = ["cityscapes_fine2_instance_seg_train"]
    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/sub_train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/sub_train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    model = InsSegModel(args=args, project_id='test', data_dir=data_dir)
    # miou = model.test()
    # print(miou)
    # model.fit()
    # model.fit_on_subset()
    losses = model.compute_loss(image_dir=image_dir,gt_dir=gt_dir)
    # for loss in losses:
    #     print(loss)
    probability = model.predict_proba(image_dir, gt_dir)
    # for prob in probability:
    #     print(prob)
    model.test()
    debug = 1
