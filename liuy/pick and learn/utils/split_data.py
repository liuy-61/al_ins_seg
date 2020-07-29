import os
from utils.add_noise import add_noise
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
from tqdm import tqdm
import cv2
from glob import glob
import csv
from sklearn.model_selection import train_test_split

img_path_all = "/home/muyun99/Desktop/supervisely/train"
mask_path_all = "/home/muyun99/Desktop/supervisely/train_mask"
ids = sorted([os.path.splitext(filename)[0] for filename in os.listdir(img_path_all) if not filename.startswith('.')])

mask_path_25 = "/home/muyun99/Desktop/supervisely/train_mask_25_pro"
csv_path_25 = "/home/muyun99/Desktop/supervisely/csv_25_pro"
mask_path_50 = "/home/muyun99/Desktop/supervisely/train_mask_50_pro"
csv_path_50 = "/home/muyun99/Desktop/supervisely/csv_50_pro"
mask_path_75 = "/home/muyun99/Desktop/supervisely/train_mask_75_pro"
csv_path_75 = "/home/muyun99/Desktop/supervisely/csv_75_pro"


# split dataset and add label_noise
def split_dataset(mask_path_save, ratio):
    if not os.path.exists(mask_path_save):
        os.mkdir(mask_path_save)
    count = 0
    flags = sorted(random.sample(ids, int(ratio * len(ids))))
    for id in tqdm(ids):
        mask_file = glob(os.path.join(mask_path_all, id) + '.*')
        mask = cv2.imread(os.path.join(mask_path_all, mask_file[0]), cv2.IMREAD_GRAYSCALE)
        if id in flags:
            count += 1
            mask = add_noise(mask)
        plt.imsave(fname=os.path.join(mask_path_save, f"{id}.png"), arr=mask, cmap=cm.gray)

    print(f'count is {count}!')


def get_csv(img_dir, mask_dir, csv_path):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    all_img_list = []
    for id in tqdm(ids):
        img_file = glob(os.path.join(img_dir, id) + '.*')
        img_file = os.path.join(img_dir, img_file[0])
        all_img_list.append(img_file)

    train_list, valid_test_list = train_test_split(all_img_list, test_size=0.2, random_state=2020)
    valid_list, test_list = train_test_split(valid_test_list, test_size=0.5, random_state=2020)

    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for item in tqdm(train_list):
        id = os.path.splitext(item.split('/')[-1])[0]
        mask_file = glob(os.path.join(mask_dir, id) + '.*')
        mask_file = os.path.join(mask_dir, mask_file[0])
        train_dict[item] = mask_file

    for item in tqdm(valid_list):
        id = os.path.splitext(item.split('/')[-1])[0]
        mask_file = glob(os.path.join(mask_path_all, id) + '.*')
        mask_file = os.path.join(mask_path_all, mask_file[0])
        valid_dict[item] = mask_file

    for item in tqdm(test_list):
        id = os.path.splitext(item.split('/')[-1])[0]
        mask_file = glob(os.path.join(mask_path_all, id) + '.*')
        mask_file = os.path.join(mask_path_all, mask_file[0])
        test_dict[item] = mask_file

    train_df = pd.DataFrame.from_dict(train_dict, orient='index')
    valid_df = pd.DataFrame.from_dict(valid_dict, orient='index')
    test_df = pd.DataFrame.from_dict(test_dict, orient='index')

    train_df.to_csv(os.path.join(csv_path, "train.csv"))
    valid_df.to_csv(os.path.join(csv_path, "valid.csv"))
    test_df.to_csv(os.path.join(csv_path, "test.csv"))


if __name__ == '__main__':
    split_dataset(mask_path_25, ratio=0.25)
    split_dataset(mask_path_50, ratio=0.50)
    split_dataset(mask_path_75, ratio=0.75)

    get_csv(img_path_all, mask_path_25, csv_path_25)
    get_csv(img_path_all, mask_path_50, csv_path_50)
    get_csv(img_path_all, mask_path_75, csv_path_75)
