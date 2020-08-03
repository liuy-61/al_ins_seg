import torch
from model import UNet_Pick, UNet, UNet_Pick_cbam
from torch.utils.data import DataLoader
from utils.dataset_pick import BasicDataset, train_transform
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from scipy import misc
import numpy as np
from evaluate import eval_net_unet_pick, eval_net_unet, eval_net_cls
import pandas as pd
from pandas.testing import assert_frame_equal
from utils.dataset_cls import BasicDataset_cls



def write_img(net):
    save_path = "/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/img_without_QAM"
    for i in range(1, 51):
        net.load_state_dict(
            torch.load(
                f"/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints/75_noise_pro_QAM_finetune/CP_epoch{i}.pth"))
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


def get_best_checkpoint(net, path, flag):
    best_score = 0
    best_epoch = None

    for i in tqdm(range(1, 51)):
        ckpt = f"{path}/CP_epoch{i}.pth"
        net.load_state_dict(torch.load(ckpt))
        if flag == "unet":
            valid_score = eval_net_unet(net, valid_dataloader, device)
        else:
            valid_score = eval_net_unet_pick(net, valid_dataloader, device)
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = ckpt

    print(f'{dir} best valid score is {best_score}')
    print(f'{dir} best valid epoch is {best_epoch}')
    return best_epoch


def get_test_score(net, best_ckpt_path, flag):
    net.load_state_dict(torch.load(best_ckpt_path))
    if flag == "unet":
        test_score = eval_net_unet(net, test_dataloader, device)
    else:
        test_score = eval_net_unet_pick(net, test_dataloader, device)
    return test_score


def get_score():
    ckpt_dir = os.listdir('/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints')
    Unet_dir = ["25_noise", "50_noise", "75_noise", "25_noise_pro", "50_noise_pro", "75_noise_pro",
                "75_noise_pro_finetune"]
    UNet_Pick_dir = ["75_noise_pro_QAM", "75_noise_pro_QAM_finetune"]
    UNet_Pick_cbam_dir = ["75_noise_pro_QAM_cbam_finetune"]
    for dir in ckpt_dir:
        print(dir)
        flag = "unet_pick"
        if dir in Unet_dir:
            net = UNet(n_classes=1, n_channels=3)
            flag = "unet"
        elif dir in UNet_Pick_dir:
            net = UNet_Pick(n_classes=1, n_channels=3)
        elif dir in UNet_Pick_cbam_dir:
            net = UNet_Pick_cbam(n_classes=1, n_channels=3)
        else:
            continue
        true_dir = os.path.join('/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints',
                                dir)
        try:
            best_ckpt_path = get_best_checkpoint(net, true_dir, flag)
            test_score = get_test_score(net, best_ckpt_path, flag)
            print(f'{dir} best test score is {test_score}')
        except Exception as ex:
            print(f'{dir} error')
            print(f'出现异常 {ex}')
            continue


def check_test_valid():
    dir = "/home/muyun99/Desktop/supervisely"
    check_list = ['csv_50_pro', 'csv_50', 'csv_25_pro', 'csv_75_pro', 'csv_75', 'csv_25']
    df_valid_list = []
    df_test_list = []
    for check_dir in check_list:
        true_dir = os.path.join(dir, check_dir)
        df_valid_list.append(pd.read_csv(os.path.join(true_dir, 'valid.csv')))
        df_test_list.append(pd.read_csv(os.path.join(true_dir, 'test.csv')))

    for df_valid in df_valid_list:
        if assert_frame_equal(df_valid, df_valid_list[0]):
            print("valid not equal!")

    for df_test in df_test_list:
        if assert_frame_equal(df_test, df_test_list[0]):
            print("test not equal!")


def find_noise_sample():
    train_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_75_pro/train.csv",
                                 transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    net = UNet_Pick(n_classes=1, n_channels=3)
    net.load_state_dict(
        torch.load(
            "/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints/75_noise_pro_QAM_finetune/CP_epoch4.pth"))
    for k, v in net.state_dict().items():
        # if 'QAM' in k or 'OCM' in k:
        print(k, v)
    # print(net.state_dict().keys())
    # print(net.state_dict()['QAM'])
    # print(net.state_dict()['OCM'])
    # net.to(device)
    # net.eval()
    #
    # # net = UNet_Pick_cbam(n_classes=1, n_channels=3)
    # # net.load_state_dict(
    # #     torch.load(
    # #         "/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints/75_noise_pro_QAM_cbam_finetune/CP_epoch4.pth"))
    # # net.to(device)
    # # net.eval()
    #
    # for batch in train_dataloader:
    #     imgs = batch['image']
    #     true_masks = batch['mask']
    #     imgs = imgs.to(device=device, dtype=torch.float32)
    #     mask_type = torch.float32 if net.Unet.n_classes == 1 else torch.long
    #     true_masks = true_masks.to(device=device, dtype=mask_type)
    #     mask_pred, weight_score = net(imgs, true_masks)
    #     print(weight_score.detach().cpu().numpy())
    #     # break


def get_submission(net, test_csv, loader, csv_save_path):
    df_test = pd.read_csv(test_csv)
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long

    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs, true_masks, label = batch['image'], batch['mask'], batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            label = label.to(device)
            mask_pred, weight_score = net(imgs, true_masks)
            probs = torch.max(torch.softmax(weight_score, dim=1), dim=1)
            probs = probs[1].cpu().numpy()
            pred_list += probs.tolist()

    print(pred_list)
    # pred_list = transform_pred(pred_list)
    submission = pd.DataFrame({
        "img_path": df_test.iloc[:, 0],
        "mask_path": df_test.iloc[:, 1],
        "true_label": df_test.iloc[:, 2],
        "pred_label": pred_list
    })
    submission.to_csv(csv_save_path, index=False, header=True)

    # return total_loss / n_val, total_acc / n_val

def test_cls():
    train_dataset = BasicDataset_cls(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/train.csv",
                                 transform=train_transform)
    val_dataset = BasicDataset_cls(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/valid.csv",
                               transform=train_transform)
    test_dataset = BasicDataset_cls(file_csv="/home/muyun99/Desktop/supervisely/csv_cls/test.csv",
                                transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 drop_last=False)
    net = UNet_Pick(n_classes=1, n_channels=3)
    net.load_state_dict(torch.load("/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints/cls_score/CP_epoch100.pth"))

    # train_loss, train_acc = eval_net_cls(net, train_dataloader, device)
    # print(f'train_loss is {train_loss}')
    # print(f'train_acc is {train_acc}')
    #
    # val_loss, val_acc = eval_net_cls(net, val_dataloader, device)
    # print(f'valid_loss is {val_loss}')
    # print(f'valid_acc is {val_acc}')
    #
    # test_loss, test_acc = eval_net_cls(net, test_dataloader, device)
    # print(f'test_loss is {test_loss}')
    # print(f'test_acc is {test_acc}')

    csv_save_path = "/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/submission/test.csv"
    get_submission(net, "/home/muyun99/Desktop/supervisely/csv_cls/test.csv", test_dataloader, csv_save_path)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8

    valid_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_75_pro/valid.csv",
                                 transform=train_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                  drop_last=True)

    test_dataset = BasicDataset(file_csv="/home/muyun99/Desktop/supervisely/csv_75_pro/test.csv",
                                transform=train_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 drop_last=True)

    # write_img()
    # get_score()
    # check_test_valid()
    # find_noise_sample()

    test_cls()