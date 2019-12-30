# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import json
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
import torch
from fvcore.common.file_io import PathManager
from fvcore.common.history_buffer import HistoryBuffer
from torch.utils.tensorboard import SummaryWriter

from detectron2.utils.events import EventWriter, _CURRENT_STORAGE_STACK


def get_event_storage():
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]



# class LiuyTensorbordPrinter(EventWriter):
#     """
#     Print **common** metrics to the terminal, including
#     iteration time, ETA, memory, all losses, and the learning rate.
#
#     To print something different, please implement a similar printer by yourself.
#     """
#
#     def __init__(self, max_iter):
#         """
#         Args:
#             max_iter (int): the maximum number of iterations to train.
#                 Used to compute ETA.
#         """
#         self.logger = logging.getLogger(__name__)
#         self._max_iter = max_iter
#
#     def write(self):
#         writer = SummaryWriter('tensorboard_log')
#         storage = get_event_storage()
#         iteration = storage.iter
#
#         data_time, time = None, None
#         eta_string = "N/A"
#         try:
#             data_time = storage.history("data_time").avg(20)
#             time = storage.history("time").global_avg()
#             eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
#             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#         except KeyError:  # they may not exist in the first few iterations (due to warmup)
#             pass
#
#         try:
#             lr = "{:.6f}".format(storage.history("lr").latest())
#         except KeyError:
#             lr = "N/A"
#
#         if torch.cuda.is_available():
#             max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
#         else:
#             max_mem_mb = None
#
#         # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
#         self.logger.info(
#             """\
# eta: {eta}  iter: {iter}  {losses}  \
# {time}  {data_time}  \
# lr: {lr}  {memory}\
# """.format(
#                 eta=eta_string,
#                 iter=iteration,
#                 losses="  ".join(
#                     [
#                         "{}: {:.3f}".format(k, v.median(20))
#                         for k, v in storage.histories().items()
#                         if "loss" in k
#                     ]
#                 ),
#                 time="time: {:.4f}".format(time) if time is not None else "",
#                 data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
#                 lr=lr,
#                 memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
#             )
#         )
#
#         loss_dic = {}
#         for k, v in storage.histories().items():
#             if "loss" in k:
#                 loss_dic[k] = v.median(20)
#                 writer.add_scalar(k, v.median(20))
#         debug =1

class LiuyTensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        for k, v in storage.histories().items():
            if "loss" in k:
                self._writer.add_scalar(k, v.median(20))

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()



