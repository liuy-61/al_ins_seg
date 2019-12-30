from abc import ABCMeta
from liuy.ins_seg2 import InsSegModel


class BaseSampler(metaclass=ABCMeta):
    def __init__(self, sampler_name, data_loader, **kwargs):
        """

        :param data_loader: select the datapoints from the data_loader
        :param sampler_name
        :param kwargs:
        """
        self.data_loader = data_loader
        self.sample_name = sampler_name

    def select_batch(self, n_sample, already_selcted, **kwargs):
        """

        :param n_sample: batch size
        :param already_selcted: index of datapoints already selected
        :param kwargs:
        :return: index of  points selected
        """
        return





