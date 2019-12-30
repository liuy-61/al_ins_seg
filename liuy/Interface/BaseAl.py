from abc import ABCMeta
from liuy.ins_seg2 import InsSegModel


class BaseAl(metaclass=ABCMeta):
    def __init__(self, seg_model, sampler, **kwargs):
        """

        :param seg_model:  model used to score the samplers.  Expects fit and predict
        methods to be implemented.
        :param sampler:
        :param kwargs:
        """
        self.seg_model = seg_model
        self.sampler = sampler

    def select_batch(self, N, already_selected,
                     **kwargs):
        pass

