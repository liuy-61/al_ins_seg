from liuy.Interface.BaseSampler import BaseSampler
from liuy.utils.ComputeLoss import LiuyComputeLoss
import re


def name2id(losses_name):
    losses_id = []
    for item in losses_name:
        loss = {}
        loss['loss_mask'] = item['loss_mask']
        digit = re.findall(r"\d+\d*", item['file_name'])
        img_id = digit[2]
        img_id = int(img_id)
        # img_id = str(img_id)
        loss['image_id'] = img_id
        losses_id.append(loss)
    return losses_id


def sort_losses(losses):
    if len(losses) <= 1:
        return losses
    pivot = losses[len(losses) // 2]
    left = [x for x in losses if x['loss_mask'] > pivot['loss_mask']]
    middle = [x for x in losses if x['loss_mask'] == pivot['loss_mask']]
    right = [x for x in losses if x['loss_mask'] < pivot['loss_mask']]
    return sort_losses(left) + middle + sort_losses(right)


class LossSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader,):
        super(LossSampler, self).__init__(sampler_name, data_loader)

    def select_batch(self, n_sample, already_selected, losses):
        losses = sort_losses(losses)
        losses = name2id(losses)
        cnt = 0
        i = 0
        samples = []
        while cnt < n_sample:
            if losses[i]['image_id'] not in already_selected and losses[i]['image_id'] not in samples:
                samples.append(losses[i]['image_id'])
                cnt += 1
            i += 1
        assert len(samples) == n_sample
        assert len(set(samples)) == len(samples)
        return samples

