import cv2
import os
import dill as pickle
import torch
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_setup, DefaultPredictor
from detectron2.config.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from liuy.Interface.BaseInsSegModel import BaseInsSegModel
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer
from liuy.utils.reg_dataset import register_a_cityscapes, register_coco_instances
from liuy.utils.torch_utils import load_prj_model
from liuy.utils.torch_utils import OUTPUT_DIR
from liuy.utils.LiuyCoCoTrainer import LiuyCoCoTrainer
from liuy.utils.ComputeLoss import LiuyComputeLoss
from liuy.utils.LiuyTrainer import LiuyTrainer

# the  config file of the model
MODEL_NAME = {
    'Faster_RCNN': '/home/tangyp/liuy/detectron2_origin/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',

    'Mask_RCNN': '/home/tangyp/liuy/detectron2_origin/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml'
    }

__all__ = ['CoCoSegModel',
           ]


# the dir where to save the model
# TRAINED_MODEL_DIR = '/media/tangyp/Data/model_file/trained_model'
# OUTPUT_DIR = '/media/tangyp/Data/model_file/OUTPUT_DIR'
class CoCoSegModel():
    """Mask_RCNN"""

    def __init__(self, args, project_id, coco_data, train_size=None, resume_or_load=False, ):
        self.args = args
        self.project_id = project_id
        self.resume_or_load = resume_or_load
        # tow ways to get model
        # 1：load the model which has been trained
        # 2：use the function：LiuyTrainer.build_model(self.cfg)
        self.model, self.device = load_prj_model(project_id=project_id)
        self.cfg = setup(args=args, project_id=project_id, coco_data=coco_data, train_size=train_size)
        if self.model is None:
            self.model = LiuyCoCoTrainer.build_model(self.cfg)
            self.model = self.model.to(self.device)
            print("Initialize a pre-trained model for project{}".format(project_id))
        else:
            print("load project {} model from file".format(project_id))
        self.trainer = LiuyCoCoTrainer(self.cfg, self.model)

    def fit(self):
        if self.resume_or_load:
            self.trainer.resume_or_load()
        self.trainer.train()
        self.save_model()

    def fit_on_subset(self, data_loader, iter_num=0):
        if self.resume_or_load:
            self.trainer.resume_or_load()
        self.trainer.data_loader = data_loader
        self.trainer._data_loader_iter = iter(data_loader)
        self.trainer.train()
        self.save_model(iter_num=iter_num)

    def fit_on_subset2(self, data_loader, iter_num, reset_model=True):
        if self.resume_or_load:
            self.trainer.resume_or_load()
        if reset_model:
            # self.model =
            del self.model
            del self.trainer
            self.model = self.model.to(self.device)
            self.trainer.model = LiuyCoCoTrainer.build_model(self.cfg).to(self.device)
        self.trainer.data_loader = data_loader
        self.trainer._data_loader_iter = iter(data_loader)
        self.trainer.train()
        self.save_model(iter_num=iter_num)

    def fit_on_subset3(self, data_loader, iter_num, reset_model=True):
        if self.resume_or_load:
            self.trainer.resume_or_load()
        if reset_model:
            del self.trainer.model
            self.trainer.model = LiuyCoCoTrainer.build_model(self.cfg).to(self.device)
            self.trainer.model.train()

            del self.trainer.optimizer
            self.trainer.optimizer = LiuyCoCoTrainer.build_optimizer(self.cfg, self.trainer.model)

            self.trainer.data_loader = data_loader
            self.trainer._data_loader_iter = iter(data_loader)

            # if comm.get_world_size() > 1:
            #     self.trainer.model = DistributedDataParallel(
            #         self.trainer.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            #     )
            del self.trainer.scheduler
            self.trainer.scheduler = LiuyCoCoTrainer.build_lr_scheduler(self.cfg, self.trainer.optimizer)

            del self.trainer.checkpointer
            self.trainer.checkpointer = DetectionCheckpointer(
                self.trainer.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.trainer.optimizer,
                scheduler=self.trainer.scheduler
            )
            self.trainer.start_iter = 0
            self.trainer.max_iter = self.cfg.SOLVER.MAX_ITER
            self.trainer.register_hooks(self.trainer.build_hooks())
        self.trainer.train()
        self.save_model(iter_num=iter_num)

    def test(self):
        """
        :return: test result on val
        """
        # self.cfg.DATASETS.TEST = ['coco_val']
        # self.cfg.DATASETS.TRAIN = ['coco_train']
        # self.trainer = LiuyCoCoTrainer(self.cfg, self.model)
        miou = self.trainer.test(self.cfg, self.trainer.model)
        return miou

    def compute_loss(self, json_file, image_root):
        """
        :param image_dir:
        :param gt_dir:
        :return: list of dict, a dict includes file_name（as a key to the data) and losses
        """
        register_coco_instances('dataset_name', json_file, image_root)
        cfg = copy.deepcopy(self.cfg)
        cfg.DATASETS.TRAIN = ["dataset_name"]
        model = copy.deepcopy(self.model)
        computer = LiuyComputeLoss(cfg, model)
        return computer.compute()

    def compute_features(self, json_file, image_root):
        pass

    def predict_proba(self, json_file, image_root, conf_thres=0.7, nms_thres=0.4,
                      verbose=True, **kwargs):
        """
                :return: list of dict, a dict includes file_name（as a key to the data) and predictions
        """
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        predictor = DefaultPredictor(self.cfg)
        register_coco_instances('dataset_name', json_file, image_root)
        data_loader, data_len = LiuyTrainer.build_test_loader(self.cfg, "dataset_name")
        results = []
        for batch in data_loader:
            for item in batch:
                file_name = item['file_name']
                img = cv2.imread(file_name)
                prediction = predictor(img)
                record = {'file_name': file_name, 'boxes': prediction['instances'].pred_boxes,
                          'labels': prediction['instances'].pred_classes, \
                          'scores': prediction['instances'].scores, 'masks': prediction['instances'].pred_masks}
                results.append(record)
        return results

    def predict(self, conf_thres=0.7):
        pass
        # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
        # predictor = DefaultPredictor(self.cfg)
        # data_loader, data_len = LiuyTrainer.build_test_loader(self.cfg, "coco_val")
        # results = []
        # for batch in data_loader:
        #     for item in batch:
        #         file_name = item['file_name']
        #         img = cv2.imread(file_name)
        #         prediction = predictor(img)
        #         img = img[:, :, ::-1]
        #         visualizer = Visualizer(img,  MetadataCatalog.get(
        #     self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        # ), instance_mode=ColorMode.IMAGE)
        #         vis_output = None
        #         if "sem_seg" in prediction:
        #              vis_output = visualizer.draw_sem_seg(
        #                 prediction["sem_seg"].argmax(dim=0).to(torch.device("cpu"))
        #             )
        #         if "instances" in prediction:
        #             instances = prediction["instances"].to(torch.device("cpu"))
        #             vis_output = visualizer.draw_instance_predictions(predictions=instances)
        #         out_filename = os.path.join(self.cfg.OUTPUT_DIR, os.path.basename(file_name))
        #         vis_output.save(out_filename)
        #         record = {'file_name': file_name, 'boxes': prediction['instances'].pred_boxes,
        #                   'labels': prediction['instances'].pred_classes, \
        #                   'scores': prediction['instances'].scores, 'masks': prediction['instances'].pred_masks}
        #         results.append(record)
        # return results

    def save_model(self, iter_num=None):
        if iter_num is not None:
            with open(os.path.join(OUTPUT_DIR, self.project_id + "_model" + str(iter_num) + ".pkl"), 'wb') as f:
                pickle.dump(self.trainer.model, f)
        else:
            with open(os.path.join(OUTPUT_DIR, self.project_id + "_model" + ".pkl"), 'wb') as f:
                pickle.dump(self.trainer.model, f)

def setup(args, project_id, coco_data, train_size=None):
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
    cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'project' + project_id)
    register_coco_instances(name='coco_train', json_file=coco_data[0]['json_file'],
                            image_root=coco_data[0]['image_root'])
    register_coco_instances(name='coco_val', json_file=coco_data[1]['json_file'], image_root=coco_data[1]['image_root'])
    cfg.DATASETS.TEST = ['coco_val']
    cfg.DATASETS.TRAIN = ['coco_train']
    if train_size is not None:
        cfg.SOLVER.MAX_ITER = int((270000 * train_size) / 45174)
        cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.78), int(cfg.SOLVER.MAX_ITER * 0.925))

    default_setup(cfg, args)
    return cfg



if __name__ == "__main__":
    coco_data = [{
                    'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                  # 'json_file': '/media/tangyp/Data/coco/annotations/sub_train2014.json',
                  'image_root': '/media/tangyp/Data/coco/train2014'
                  },
                 {
                     # 'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                     'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',
                     'image_root': '/media/tangyp/Data/coco/val2014'
                 }]
    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='random_100', coco_data=coco_data, resume_or_load=True)
    # seg_model.fit()
    # seg_model.trainer.resume_or_load()
    seg_model.test()
    # miou=seg_model.test()
    # model.predict()
    #prediction = model.predict_proba(coco_data[1]['json_file'], coco_data[1]['image_root'])
    debug = 1