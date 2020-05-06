import random

from liuy.Interface.BaseSampler import BaseSampler

# class CityRandomSampler(BaseSampler):
#     def __init__(self, sampler_name, data_loader):
#         super(CityRandomSampler, self).__init__(sampler_name, data_loader)
#         self.image_files_list = []
#         lt = data_loader.dataset._dataset._lst
#         # file_name as key to data
#         for item in lt:
#             self.image_files_list.append(item['file_name'])
#
#     def select_batch(self, n_sample, already_selected):
#         cnt = 0
#         samples = []
#         while cnt < n_sample:
#             sample = random.sample(self.image_files_list, 1)
#             if sample[0] not in already_selected and sample[0] not in samples:
#                 cnt += 1
#                 samples.append(sample[0])
#         assert len(samples) == n_sample
#         return samples

class CoCoRandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(CoCoRandomSampler, self).__init__(sampler_name, data_loader)

    def select_batch(self, n_sample, already_selected):
        cnt = 0
        samples = []
        while cnt < n_sample:
            sample = random.sample(self.image_files_list, 1)
            if sample[0] not in already_selected and sample[0] not in samples:
                cnt += 1
                samples.append(sample[0])

        assert len(samples) == n_sample
        assert len(set(samples)) == len(samples)
        return samples