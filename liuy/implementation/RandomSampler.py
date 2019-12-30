import random

from liuy.Interface.BaseSampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(RandomSampler, self).__init__(sampler_name, data_loader)

        self.image_files_list = []
        lt = data_loader.dataset._dataset._lst
        for item in lt:
            self.image_files_list.append(item['file_name'])


    def select_batch(self, n_sample, already_selected):
        cnt = 0
        samples = []
        while cnt < n_sample:
           sample = random.sample(already_selected, 1)
           if sample not in already_selected:
               cnt += 1
               samples.append(sample)
        assert len(samples) == n_sample
        return samples
