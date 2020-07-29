import numpy as np
import cv2
import random


def add_noise(img_array):
    kernel = np.ones((3, 3), np.uint8)
    if random.randint(0, 1) == 0:
        img_noise = cv2.erode(img_array, kernel, iterations=6)
    else:
        img_noise = cv2.dilate(img_array, kernel, iterations=6)
    _, img_noise = cv2.threshold(img_noise, 1, 255, cv2.THRESH_BINARY)
    return img_noise


if __name__ == '__main__':
    img = cv2.imread("/home/muyun99/Desktop/supervisely/train_mask/2767.png")
    # img_noise_erosion, img_noise_dilation = add_noise(img)
    # cv2.imshow("img_noise_dilation", img_noise_dilation)
    # cv2.imshow("img_noise_erosion", img_noise_erosion)

    img_noise = add_noise(img)

    cv2.imshow("img", img)
    cv2.imshow("img_noise_", img_noise)

    cv2.waitKey(0)
