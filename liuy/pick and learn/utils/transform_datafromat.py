# 将supervisely人像数据集的格式转为unet格式
import supervisely_lib as sly
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cv2
import os

img_savepath = "/home/muyun99/Desktop/supervisely/train"
mask_savepath = "/home/muyun99/Desktop/supervisely/train_mask"
sly_path = "7/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/Supervisely Person Dataset"


def display_images(images, img_idx, figsize=None):
    plt.figure(figsize=(figsize if (figsize is not None) else (15, 15)))
    for i, img in enumerate(images, start=1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)


def show_label():
    project = sly.Project(sly_path, sly.OpenMode.READ)
    # 打印数据集相关信息
    print("Project name: ", project.name)
    print("Project directory: ", project.directory)
    print("Total images: ", project.total_items)
    print("Dataset names: ", project.datasets.keys())

    img_idx = 1

    for dataset in project:
        for item_name in dataset:
            img_path = dataset.get_img_path(item_name)
            item_paths = project.datasets.get(dataset.name).get_item_paths(img_path.split('/')[-1])

            # 读取原始图像和标注信息
            img = sly.image.read(item_paths.img_path)
            ann = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)

            # 使用meta.json文件中的颜色呈现所有标签。
            ann_render = np.zeros(ann.img_size + (3,), dtype=np.uint8)
            ann.draw(ann_render)

            gray_img = cv2.cvtColor(ann_render, cv2.COLOR_RGB2GRAY)

            _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

            dim = (572, 572)
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
            # 显示

            plt.imsave(fname=os.path.join(img_savepath, f"{img_idx}.png"), arr=resized_img)
            plt.imsave(fname=os.path.join(mask_savepath, f"{img_idx}.png"), arr=resized_mask, cmap=cm.gray)
            # display_images([resized_img, resized_mask], img_idx)
            img_idx += 1
            # break



def transform_format():
    pass


if __name__ == '__main__':
    show_label()
    transform_format()
