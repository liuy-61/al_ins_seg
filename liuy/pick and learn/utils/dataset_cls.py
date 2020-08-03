from torch.utils.data import Dataset
import logging
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import torch
import numpy as np

class BasicDataset_cls(Dataset):
    def __init__(self, file_csv, transform):
        super(BasicDataset_cls, self).__init__()
        self.df = pd.read_csv(file_csv)
        # print(self.df.head())
        # print(self.df.iloc[1, 0])
        self.transform = transform
        logging.info(f'Creating dataset with {len(self.df)} examples')

    def __getitem__(self, index):
        # print(self.df.iloc[index, 0])
        # print(self.df.iloc[index, 1])

        img = Image.open(self.df.iloc[index, 0]).convert('RGB')
        img = self.transform(img)

        mask = Image.open(self.df.iloc[index, 1]).convert('L')
        mask = self.transform(mask)

        label = int(self.df.iloc[index, 2])

        return {
            'image': img, 
            'mask': mask,
            'label': torch.from_numpy(np.array(label, dtype=np.int64))
        }

    def __len__(self):
        return len(self.df)


train_transform = transforms.Compose([
    transforms.Resize([321, 321]),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    batch_size = 8

    train_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/train.csv",
                                 transform=train_transform)
    val_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/valid.csv",
                               transform=train_transform)
    test_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/test.csv",
                                transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 drop_last=True)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    for batch in train_dataloader:
        imgs = batch['image']
        true_masks = batch['mask']
        label = batch['label']
        print("单个img的size: ", imgs.shape)
        print("单个mask的size: ", true_masks.shape)
        print("当前sample的label: ", label)


        assert imgs.shape[1] == 3, \
            f'Network has been defined with 3 input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
