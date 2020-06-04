import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import config

class MyDataSet(Dataset):
    def __init__(self, df, transform, mode='train'):
        super(MyDataSet, self).__init__()
        self.df = df
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            try:
                img = Image.open(self.df['filename'].iloc[index]).convert('RGB')
                img = self.transform(img)
                return img, torch.from_numpy(np.array(self.df['label'].iloc[index])), int(
                    self.df['filename'].iloc[index][-16:-4])
                # return img, 1, 1
            except:
                print("load file error!")
                print(index)
                print(self.df['filename'].iloc[index])
        else:
            img = Image.open(self.df[index]).convert('RGB')
            img = self.transform(img)
            return img, torch.from_numpy(np.array(0)), int(self.df['filename'].iloc[index][-16:-4])

    def __len__(self):
        return len(self.df)


transform = transforms.Compose([
    transforms.Resize([config.IMAGE_SIZE, config.IMAGE_SIZE]),
    transforms.ToTensor(),
])
