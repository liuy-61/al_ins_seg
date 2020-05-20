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


def decrease_sort_losses(losses):
    if len(losses) <= 1:
        return losses
    pivot = losses[len(losses) // 2]
    left = [x for x in losses if x['loss_mask'] > pivot['loss_mask']]
    middle = [x for x in losses if x['loss_mask'] == pivot['loss_mask']]
    right = [x for x in losses if x['loss_mask'] < pivot['loss_mask']]
    return decrease_sort_losses(left) + middle + decrease_sort_losses(right)

def increase_sort_losses(losses):
    if len(losses) <= 1:
        return losses
    pivot = losses[len(losses) // 2]
    left = [x for x in losses if x['loss_mask'] < pivot['loss_mask']]
    middle = [x for x in losses if x['loss_mask'] == pivot['loss_mask']]
    right = [x for x in losses if x['loss_mask'] > pivot['loss_mask']]
    return increase_sort_losses(left) + middle + increase_sort_losses(right)


class LossSampler():
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name

    def select_batch(self, n_sample, already_selected, losses, loss_decrease=True):
        if loss_decrease:
            losses = decrease_sort_losses(losses)
        else:
            losses = increase_sort_losses(losses)

        cnt = 0
        i = 0
        samples = []
        while cnt < n_sample:
            if losses[i]['image_id'] not in already_selected and losses[i]['image_id'] not in samples:
                samples.append(losses[i]['image_id'])
                cnt += 1
                if cnt%10 == 0:
                    print("has got {} image id, still need {} image id".format(cnt, n_sample-cnt))
            i += 1
        # while cnt < n_sample:
        #     for loss in losses:
        #         if loss['image_id'] not in already_selected and loss['image_id'] not in samples:
        #             samples.append(loss['image_id'])
        #             cnt += 1
        assert len(samples) == n_sample
        assert len(set(samples)) == len(samples)
        return samples

if __name__ == '__main__':
    pass