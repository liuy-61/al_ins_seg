import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def add_noise(img_array):
    kernel = np.ones((3, 3), np.uint8)
    if random.randint(0, 1) == 0:
        img_noise = cv2.erode(img_array, kernel, iterations=6)
    else:
        img_noise = cv2.dilate(img_array, kernel, iterations=6)
    _, img_noise = cv2.threshold(img_noise, 1, 255, cv2.THRESH_BINARY)
    return img_noise

def add_noise_iteration(img_array, iterations):
    kernel = np.ones((3, 3), np.uint8)
    if random.randint(0, 1) == 0:
        img_noise = cv2.erode(img_array, kernel, iterations=iterations)
    else:
        img_noise = cv2.dilate(img_array, kernel, iterations=iterations)
    _, img_noise = cv2.threshold(img_noise, 1, 255, cv2.THRESH_BINARY)
    return img_noise

def show_noise(img_array, iterations, name):
    kernel = np.ones((3, 3), np.uint8)

    for index, iteration in enumerate(iterations):
        img_noise_erode = cv2.erode(img_array, kernel, iterations=iteration)
        img_noise_dilate = cv2.dilate(img_array, kernel, iterations=iteration)
        _, img_noise_erode = cv2.threshold(img_noise_erode, 1, 255, cv2.THRESH_BINARY)
        _, img_noise_dilate = cv2.threshold(img_noise_dilate, 1, 255, cv2.THRESH_BINARY)
        print(index)
        plt.subplot(len(iterations), 3, 3 * index + 1)
        plt.imshow(img_array)
        plt.title("img", fontsize=8)

        plt.subplot(len(iterations), 3, 3 * index + 2)
        plt.imshow(img_noise_erode)
        plt.title(f"img_noise_erode_{iteration}", fontsize=8)

        plt.subplot(len(iterations), 3, 3 * index + 3)
        plt.imshow(img_noise_dilate)
        plt.title(f"img_noise_dilate_{iteration}", fontsize=8)

    # plt.savefig(os.path.join("/media/muyun99/DownloadResource/onedrive/研究生事宜/实验室资料/7.阅读笔记/pic", f"{name}_show_noise.png"))
    plt.show()


if __name__ == '__main__':
    path = "/home/muyun99/Desktop/supervisely/train_mask/136.png"
    img = cv2.imread(path)
    # img_noise_erosion, img_noise_dilation = add_noise(img)
    # cv2.imshow("img_noise_dilation", img_noise_dilation)
    # cv2.imshow("img_noise_erosion", img_noise_erosion)

    # img_noise = add_noise(img)
    show_noise(img, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name=path.split('/')[-1].split('.')[0])

    # cv2.imshow("img", img)
    # cv2.imshow("img_noise_", img_noise)
    # cv2.waitKey(0)
