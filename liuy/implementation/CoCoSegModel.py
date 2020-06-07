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
from liuy.utils.reg_dataset import register_a_cityscapes, register_coco_instances, \
    register_coco_instances_from_selected_image_files
from liuy.utils.torch_utils import load_prj_model, select_device
from liuy.utils.torch_utils import OUTPUT_DIR
from liuy.utils.LiuyCoCoTrainer import LiuyCoCoTrainer
from liuy.utils.ComputeLoss import LiuyComputeLoss
from liuy.utils.LiuyTrainer import LiuyTrainer
from liuy.utils.LiuyFeatureGetter import LiuyFeatureGetter
from liuy.utils.local_config import coco_data, debug_data, MODEL_NAME
from liuy.utils.K_means import read_image2class
from liuy.implementation.LossSampler import LossSampler
# the  config file of the model


__all__ = ['CoCoSegModel',
           ]


# the dir where to save the model
# TRAINED_MODEL_DIR = '/media/tangyp/Data/model_file/trained_model'
# OUTPUT_DIR = '/media/tangyp/Data/model_file/OUTPUT_DIR'
class CoCoSegModel():
    """Mask_RCNN"""

    def __init__(self, args,
                 project_id,
                 coco_data,
                 model_config='Mask_RCNN',
                 train_size=None,
                 resume_or_load=False,):
        """

        :param args:
        :param project_id:
        :param coco_data:
        :param model_config: the default is Mask_RCNN without FPN, if need the Mask_RCNN with FPN then specify the config_file
        as 'Mask_RCNN2'
        :param train_size:
        :param resume_or_load:
        """
        self.args = args
        self.project_id = project_id
        self.resume_or_load = resume_or_load
        self.coco_data = coco_data
        self.model_config = model_config
        self.device = select_device()
        # three ways to get model
        # 1：load the model which has been trained
        # 2：use the function：LiuyTrainer.build_model(self.cfg)
        # 3:from the parameter
        self.model, self.device = load_prj_model(project_id=project_id)
        self.cfg = setup(args=args,
                         project_id=project_id,
                         coco_data=coco_data,
                         model_config=self.model_config,
                         train_size=train_size)

        if self.model is None:
            self.model = LiuyCoCoTrainer.build_model(self.cfg)
            self.model = self.model.to(self.device)
            print("Initialize a pre-trained model for project {}".format(project_id))
        else:
            print("load project {} model from file".format(project_id))
        self.trainer = LiuyCoCoTrainer(self.cfg, self.model)

    def reset_model(self):
        """
        reset the model in CoCoSegModel and the model in LiuyCoCoTrainer
        reset the cfg
        :return:
        """
        del self.cfg
        self.cfg = setup(args=self.args,
                         project_id=self.project_id,
                         coco_data=self.coco_data,
                         model_config=self.model_config)

        del self.model
        self.model = LiuyCoCoTrainer.build_model(self.cfg)
        print("initialize a new model for project {} ".format(self.project_id))
        self.model = self.model.to(self.device)
        self.trainer.reset_model(cfg=self.cfg, model=self.model)

    def set_model(self, model, creat_new_folder=False):
        """
        del the self.model and then set the parameter model as self.model
        :param model:
        :return:
        """
        del self.cfg
        self.cfg = setup(args=self.args,
                         project_id=self.project_id,
                         coco_data=self.coco_data,
                         create_new_folder=creat_new_folder,
                         model_config=self.model_config)

        del self.model
        self.model = model
        print("initialize a new model for project {} ".format(self.project_id))
        self.model = self.model.to(self.device)
        self.trainer.reset_model(cfg=self.cfg, model=self.model)

    def back_to_base_model(self, base_model):
        """

        :param base_model:
        :return:
        """
        del self.cfg
        self.cfg = setup(args=self.args,
                         project_id=self.project_id,
                         coco_data=self.coco_data,
                         model_config=self.model_config,
                         create_new_folder=None)

        del self.model
        self.model = copy.deepcopy(base_model)
        # print("initialize a new model for project {} ".format(self.project_id))
        self.model = self.model.to(self.device)
        self.trainer.reset_model(cfg=self.cfg, model=self.model)

    def fit(self):
        if self.resume_or_load:
            self.trainer.resume_or_load()
        data_len = len(self.trainer.data_loader.dataset._dataset._lst)
        self.trainer.max_iter = int((270000 * data_len) / 45174)
        self.trainer.train()

        self.save_model()

    def fit_on_subset(self, data_loader, iter_num=0):
        if self.resume_or_load:
            self.trainer.resume_or_load()

        self.trainer.data_loader = data_loader
        self.trainer._data_loader_iter = iter(data_loader)
        data_len = len(data_loader.dataset._dataset._lst)
        self.trainer.max_iter = int((270000 * data_len) / 45174)
        self.trainer.train()
        self.save_model(iter_num=iter_num)

    def fit_on_single_data(self, image_id_list):
        """
            for each image in image_id_list build a data_loader iteratively
            return a list of dict,dict {'image_id':int, 'score':float}
            use every data in image_id_list to fine tuning the base model
            and compute the promotion as the image's score
        """
        score_list = []

        base_model = copy.deepcopy(self.model)
        result = self.test()
        base_score = result['segm']['AP']

        for image_id in image_id_list:
            dic = {'image_id': image_id}
            image_id = [image_id]
            register_coco_instances_from_selected_image_files(name='coco_from_selected_image',
                                                              json_file=coco_data[0]['json_file'],
                                                              image_root=coco_data[0]['image_root'],
                                                              selected_image_files=image_id)

            data_loader, l = self.trainer.re_build_train_loader('coco_from_selected_image',
                                                                images_per_batch=1)

            self.trainer.data_loader = data_loader
            self.trainer._data_loader_iter = iter(data_loader)
            self.trainer.max_iter = 20
            result = self.trainer.train()
            dic['score'] = result['segm']['AP'] - base_score
            score_list.append(dic)

            # back_to_base model
            self.back_to_base_model(base_model=base_model)

        # save score_list
        self.save_score_list(score_list)
        return score_list

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

    def save_mask_features(self, json_file, image_root, selected_image_file):
        """

        :param selected_image_file: a list of image id,when save mask features we split
        selected and unselected images' mask feature
        :param json_file:  coco data's json_file
        :param image_root: coco data's image_root
        use the json_file & image_root to build data_loader and then extract mask_features from it
        a list of dict, dict :{'image_id':int, 'feature_tensor':tensor}
        the tensors shape is (M,C,output_size,output_size)
        """
        register_coco_instances('dataset_name', json_file, image_root)
        cfg = copy.deepcopy(self.cfg)
        cfg.DATASETS.TRAIN = ["dataset_name"]
        model = copy.deepcopy(self.model)
        getter = LiuyFeatureGetter(cfg, model)
        getter.save_feature(self.project_id, selected_image_file=selected_image_file)

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
        """

        :param iter_num:
        :return: save the model the path same as log
        """
        self.cfg.OUTPUT_DIR
        with open(os.path.join(self.cfg.OUTPUT_DIR, self.project_id + "_model" + ".pkl"), 'wb') as f:
            pickle.dump(self.trainer.model, f)

    def save_selected_image_id(self, selected_image_id):
        """

        :param selected_image_id:
        :return: None
        save the selected image id to dir which is same as the model
        """
        detail_output_dir = os.path.join(self.cfg.OUTPUT_DIR, 'selected_image_id')
        with open(detail_output_dir + '.pkl', 'wb') as f:
            pickle.dump(selected_image_id, f, pickle.HIGHEST_PROTOCOL)
        print("save img_id_list successfully")

    def save_score_list(self, score_list):
        """

        :param score_list: a list of dict,dict {'image_id':int, 'score':float}
        :return: None
        save the score_list
        """
        detail_output_dir = os.path.join(OUTPUT_DIR, 'file')
        if not os.path.exists(detail_output_dir):
            os.makedirs(detail_output_dir)
        detail_file = os.path.join(detail_output_dir, 'score_list.pkl')

        # if the score_list has saved before, read it and merge it with new score_list
        with open(detail_file, 'rb') as f:
            old = pickle.load(f)
        if old is not None:
            score_list.extend(old)

        with open(detail_file, 'wb') as f:
            pickle.dump(score_list, f, pickle.HIGHEST_PROTOCOL)
        print("save score_list successfully")

    def read_selected_image_id(self, iteration=None):
        """

        :param iteration:
        :return:
        """
        if iteration is None:
            detail_output_dir = os.path.join(self.cfg.OUTPUT_DIR, 'selected_image_id')
        with open(detail_output_dir + '.pkl', 'rb') as f:
            return pickle.load(f)


def setup(args, project_id, coco_data, model_config, create_new_folder=True, train_size=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    config_file = MODEL_NAME[model_config]
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.MASK_ON = True
    if create_new_folder:
        folder_num = get_folder_num(project_id=project_id)
        cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'project_' + project_id, folder_num)
    else:
        cfg.OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'project_' + project_id)
    register_coco_instances(name='coco_train', json_file=coco_data[0]['json_file'],
                            image_root=coco_data[0]['image_root'])
    register_coco_instances(name='coco_val', json_file=coco_data[1]['json_file'],
                            image_root=coco_data[1]['image_root'])
    cfg.DATASETS.TEST = ['coco_val']
    cfg.DATASETS.TRAIN = ['coco_train']
    if train_size is not None:
        cfg.SOLVER.MAX_ITER = int((270000 * train_size) / 45174)
        cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.78), int(cfg.SOLVER.MAX_ITER * 0.925))

    default_setup(cfg, args)
    return cfg

def get_folder_num(project_id):
    """
    :return:get folder num next should be
    """
    iter = 0
    dir = os.path.join(OUTPUT_DIR, 'project_' + project_id)

    while True:
        folder = dir + '/' + str(iter)
        if os.path.exists(folder) == True:
            iter += 1
        else:
            break
    return str(iter)


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='coco',
                             coco_data=coco_data,
                             model_config='Mask_RCNN',
                             resume_or_load=True)
    seg_model.save_mask_features(json_file=coco_data[0]['json_file'],
                                 image_root=coco_data[0]['image_root'],
                                 selected_image_file=[])

