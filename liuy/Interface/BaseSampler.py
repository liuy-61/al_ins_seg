from abc import ABCMeta
from liuy.ins_seg2 import InsSegModel


class BaseSampler(metaclass=ABCMeta):
    def __init__(self, sampler_name, data_loader, **kwargs):
        """

        :param data_loader: we use the data_loader to init a  image_files_list, then we select data from image_files_list.
        :param sampler_name
        :param kwargs:
        """
        self.data_loader = data_loader
        self.sample_name = sampler_name
        self.image_files_list = []
        lt = data_loader.dataset._dataset._lst
        # file_name as key to data
        for item in lt:
            self.image_files_list.append(item['file_name'])

    def select_batch(self, n_sample, already_selected, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param already_selected: list of file_name already selected
        :param kwargs:
        :return: list of file_name you selected this batch
        """
        return





