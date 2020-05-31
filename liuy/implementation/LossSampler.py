import pandas as pd

from liuy.Interface.BaseSampler import BaseSampler
from liuy.utils.ComputeLoss import LiuyComputeLoss
import copy
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

    def group_image(slef, image2class):
        """

        :param image2class: a nd array, m * 2, m means number of feature,
        the first column is the image id, and the second is the class id
        :return: list of list, each list is contain the same class' image id
        """
        image2class_pd = pd.DataFrame(image2class)
        groups = []
        a = image2class_pd.groupby(1)
        for i in range(a.ngroups):
            group = image2class[a.groups[i]][:, 0]
            groups.append(group)
        return groups

    def group_loss(self, image_groups, losses):
        """

        :param image_groups: list of array, each array is contain the same class' image id
        :param losses: list of dict, dict: 'image_id': int 'mask_loss':tensor
        :return: list o list , each child list is list of dict, dict: 'image_id': int 'mask_loss':tensor
        the dict in same child list is in same class
        """
        loss_groups = []
        for i, array in enumerate(image_groups):
            loss_group = [loss_dict for loss_dict in losses if loss_dict['image_id'] in array]
            loss_groups.append(loss_group)
            print('complete {} groups'.format(i))

        return loss_groups

    def select_batch(self, n_sample, already_selected, losses, loss_decrease=False):
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

    def slect_batch_from_groups(self,
                                image2class,
                                n_sample,
                                already_selected,
                                losses,
                                loss_decrease=False):

        image_group = self.group_image(image2class)

        loss_group = self.group_loss(image_groups=image_group,
                                     losses=losses)

        # clear the loss_group, remove the item in already_selected
        # compute each group's number and total number
        group_len = []
        total_len = 0

        for i, group in enumerate(loss_group):
            group = [loss_dict for loss_dict in group if loss_dict['image_id'] not in already_selected]
            group_len.append(len(group))
            total_len += len(group)

        # sample from each group in proportion
        samples = []
        for i, group in enumerate(loss_group):
            amount = int((group_len[i] / total_len) * n_sample)
            if amount > group_len[i]:
                amount = group_len[i]

            # use loss_sampler sample amount image id from this group
            print("select from {}th group".format(i))
            sample = self.select_batch(n_sample=amount,
                                        already_selected=already_selected,
                                        losses=group,
                                        loss_decrease=False)
            already_selected.extend(sample)
            samples.extend(sample)


        # check the amount of samples whether reach n_sample, if not sample from losses
        samples_len = len(samples)
        residue = n_sample - samples_len
        if residue > 0:
            print('select from last iter')
            sample = self.select_batch(n_sample=residue,
                                        already_selected=already_selected,
                                        losses=losses,
                                        loss_decrease=False)
            samples.extend(sample)
        assert len(samples) == n_sample
        return samples



if __name__ == '__main__':
    a = [2,3,4,5,6,7]
    b = []
    a.extend(b)
    debug = 1