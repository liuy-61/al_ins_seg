from detectron2.engine.train_loop import HookBase
# from detectron2.reg_dataset1 import get_custom_dicts
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer
import argparse
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel
from detectron2.config.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
from detectron2.engine import hooks
from liuy.utils.save_mask_feature import save_mask_feature

# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import time
import weakref
import torch

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

__all__ = ["FeatureGetterBase"]

"""
    FeatureGetterBase imitate the class TrainerBase in train_loop.py

"""


class FeatureGetterBase:
    """
    Base class for iterative feature getter with hooks.

    The only assumption we made here is: the getting feature runs in a loop.
    A subclass can implement what the loop is.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end getting feature.

        storage(EventStorage): An EventStorage that's opened during the course of getting feature.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the feature getter. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and feature getter cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def save_feature(self, start_iter: int, max_iter: int, project_id=None, selected_image_file=None):
        """
        Args:
            start_iter, max_iter (int): See docs above
            return a list of dict, dict :{'image_id':int, 'feature_tensor':tensor}
            to avoid out of memory, we save the feature_list as file,
            and the project_id is used to get the path where to save,
            selected_image_file : a list of image id, it is used to split the selected images' mask feature
            and unselected images' mask feature

        """
        logger = logging.getLogger(__name__)
        logger.info("Starting getting feature from iteration {}".format(start_iter))

        selected_feature_list = []
        unselected_feature_list = []
        selected_serial_number = 0
        unselected_serial_number = 0

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    """
                    in self.run step do what is diff
                    """
                    len_slected_feature = 20
                    feature = self.run_step()
                    if project_id is not None and selected_image_file is not None:

                        if feature['image_id'] in selected_image_file:
                            selected_feature_list.append(feature)
                            if len(selected_feature_list) == len_slected_feature:
                                save_mask_feature(project_id=project_id, mask_feature=selected_feature_list,
                                                  serial_number=selected_serial_number, selected_or_not=True)

                                selected_serial_number += 1
                                num = selected_serial_number * len_slected_feature + unselected_serial_number * len_slected_feature
                                print("save {}  images' mask feature, still need {} images' feature to save "
                                      .format(num, self.max_iter-num))
                                del selected_feature_list
                                selected_feature_list = []

                        else:
                            unselected_feature_list.append(feature)
                            if len(unselected_feature_list) == len_slected_feature:
                                save_mask_feature(project_id=project_id, mask_feature=unselected_feature_list,
                                                  serial_number=unselected_serial_number, selected_or_not=False)

                                unselected_serial_number += 1
                                num = selected_serial_number * len_slected_feature + unselected_serial_number * len_slected_feature
                                print("save {}  images' mask feature, still need {} images' feature to save "
                                      .format(num, self.max_iter - num))
                                del unselected_feature_list
                                unselected_feature_list = []


                    self.after_step()
            finally:
                self.after_train()
                if len(selected_feature_list) > 0:
                    save_mask_feature(project_id=project_id, mask_feature=selected_feature_list,
                                      serial_number=selected_serial_number, selected_or_not=True)
                    selected_serial_number += 1

                if len(unselected_feature_list) > 0:
                    save_mask_feature(project_id=project_id, mask_feature=unselected_feature_list,
                                      serial_number=unselected_serial_number, selected_or_not=False)

                    unselected_serial_number += 1


    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleFeatureGetter(FeatureGetterBase):
    """
    A simple feature getter for the most common type of task:
    It assumes that every step, you:

    1. Compute the feature with a data from the data_loader.


    If you want to do anything fancier than this,
    either subclass FeatureGetterBase and implement your own `run_step`,
    or write your own  loop.
    """

    def __init__(self, model, data_loader):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.

        """
        super().__init__()

        """
        We set the model to training mode in the feature getter.
        but we do not optimize the model
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

    def run_step(self):
        """
        Implement the standard getting feature logic described above.
        """
        # assert self.model.training, "[SimpleFeatureGetter] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the data loader.
        return a dict 'image_id' is the feature's corresponding image' index
        """
        data = next(self._data_loader_iter)

        with torch.no_grad():
            """
            If your want to do something with the losses, you can wrap the model.
            """
            assert len(data) == 1, 'batch_size is not 1'
            feature_dict = {'image_id': data[0]['image_id'], 'feature_tensor': self.model.get_mask_feature(data)}
            return feature_dict


class LiuyFeatureGetter(SimpleFeatureGetter):
    def __init__(self, cfg, model=None):

        if model is not None:
            model = model
        else:
            model = self.build_model(cfg)
        data_loader = self.build_train_loader(cfg)
        super().__init__(model=model, data_loader=data_loader)
        self.max_iter = len(data_loader.dataset._dataset._lst)
        self.cfg = cfg
        self.start_iter = 0
        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.

        return ret

    def save_feature(self, project_id=None, selected_image_file=None):
        """
        project_id: to avoid out of memory  we save the feature part by part ,
        and the project_id is used to compute the path to save
        Returns:
            OrderedDict of results,
        """

        feature = super().save_feature(self.start_iter, self.max_iter, project_id=project_id,
                                       selected_image_file=selected_image_file)
        return feature

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)
