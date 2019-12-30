import random

from liuy.Interface.BaseSampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(RandomSampler, self).__init__(sampler_name, data_loader)


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
