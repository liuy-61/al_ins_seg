import torch
from model import UNet_Pick
from torch.utils.data import DataLoader
from utils.dataset_pick import BasicDataset, train_transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from scipy import misc
import numpy as np
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_Pick(n_classes=1, n_channels=3)
    # for k, v in net.state_dict().items():
    #     print(k)
    batch_size = 1
    test_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_75_pro/test.csv",
                                 transform=train_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 drop_last=True)
    save_path = "/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/img_without_QAM"
    for i in range(1, 51):

        net.load_state_dict(
            torch.load(f"/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints/75_noise_pro_QAM_finetune/CP_epoch{i}.pth"))
        idx = 0
        net.eval()
        net.to(device)
        img_dir = os.path.join(save_path, str(i))
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        for batch in tqdm(test_dataloader):
            imgs = batch['image']
            true_masks = batch['mask']
            assert imgs.shape[1] == net.Unet.n_channels, \
                'Network has been defined with {} input channels, '.format(
                    net.n_channels) + 'but loaded images have {} channels. Please check that '.format(
                    imgs.shape[1]) + 'the images are loaded correctly.'


            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred, weight_score = net(imgs, true_masks)
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float().cpu().numpy()
            mask_pred *= 255
            mask_pred = np.transpose(mask_pred[0], (1, 2, 0))
            path = os.path.join(img_dir, f'{idx}.png')
            # print(type(mask_pred))
            # print(mask_pred.shape)
            # print(path)
            idx += 1
            cv2.imwrite(filename=path, img=mask_pred)

            # print()









