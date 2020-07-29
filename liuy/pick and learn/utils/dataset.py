import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader, random_split

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super(BasicDataset, self).__init__()

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.ids = [os.path.splitext(filename)[0] for filename in os.listdir(imgs_dir) if not filename.startswith('.')]

        # self.ids = glob(f'{imgs_dir}*/img/*')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, scale):
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = img.resize((newW, newH))
        img_array = np.array(img)

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        img_trans = img_array.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        img_file = self.ids[i]
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        mask = Image.open(mask_file[0])

        mask = mask.convert('L')
        img = Image.open(img_file[0])
        img = img.convert("RGB")
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, scale=self.scale)
        mask = self.preprocess(mask, scale=self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
if __name__ == '__main__':
    dir_img = "/home/muyun99/Desktop/supervisely/train/"
    dir_mask = "/home/muyun99/Desktop/supervisely/train_mask/"
    scale = 1.0
    val_percent = 0.1
    batch_size = 8


    dataset = BasicDataset(imgs_dir=dir_img, masks_dir=dir_mask, scale=scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                drop_last=True)

    for batch in train_dataloader:
        imgs = batch['image']
        true_masks = batch['mask']
        print(imgs.shape)
        print(true_masks.shape)
        assert imgs.shape[1] == 3, \
            f'Network has been defined with 3 input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'