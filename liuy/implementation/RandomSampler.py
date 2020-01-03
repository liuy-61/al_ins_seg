import random

from liuy.Interface.BaseSampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(RandomSampler, self).__init__(sampler_name, data_loader)


    def select_batch(self, n_sample, already_selected):
        cnt = 0
        samples = []
        while cnt < n_sample:
            sample = random.sample(self.image_files_list, 1)
            if sample[0] not in already_selected:
                cnt += 1
                samples.append(sample[0])
        assert len(samples) == n_sample
        return samples
