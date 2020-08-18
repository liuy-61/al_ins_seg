import cv2
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
major = cv2.__version__.split('.')[0]  # Get opencv version
bDebug = False

""" For precision, contours_a==GT & contours_b==Prediction
    For recall, contours_a==Prediction & contours_b==GT """


def calc_precision_recall(contours_a, contours_b, threshold):
    top_count = 0

    try:
        for b in range(len(contours_b)):

            # find the nearest distance
            for a in range(len(contours_a)):
                dist = (contours_a[a][0] - contours_b[b][0]) * \
                       (contours_a[a][0] - contours_b[b][0])
                dist = dist + \
                       (contours_a[a][1] - contours_b[b][1]) * \
                       (contours_a[a][1] - contours_b[b][1])
                if dist < threshold * threshold:
                    top_count = top_count + 1
                    break

        precision_recall = top_count / len(contours_b)
    except Exception as e:
        precision_recall = 0

    return precision_recall, top_count, len(contours_b)


""" computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation """


def bfscore(gtfile, prfile, threshold=2):
    gt__ = cv2.imread(gtfile)  # Read GT segmentation
    gt_ = cv2.cvtColor(gt__, cv2.COLOR_BGR2GRAY)  # Convert color space

    pr_ = cv2.imread(prfile)  # Read predicted segmentation
    pr_ = cv2.cvtColor(pr_, cv2.COLOR_BGR2GRAY)  # Convert color space

    classes_gt = np.unique(gt_)  # Get GT classes
    classes_pr = np.unique(pr_)  # Get predicted classes

    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        # print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)

        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
        # print('Merged classes :', classes)
    else:
        # print('Classes :', classes_gt)
        classes = classes_gt  # Get matched classes

    m = np.max(classes)  # Get max of classes (number of classes)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros((m + 1), dtype=float)
    areas_gt = np.zeros((m + 1), dtype=float)

    for i in range(m + 1):
        bfscores[i] = np.nan
        areas_gt[i] = np.nan

    for target_class in classes:  # Iterate over classes

        if target_class == 0:  # Skip background
            continue

        # print(">>> Calculate for class:", target_class)

        gt = gt_.copy()
        gt[gt != target_class] = 0
        # print(gt.shape)

        # contours는 point의 list형태.
        if major == '3':  # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape
        else:  # For other opencv versions
            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

        # contours 는 list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())
        if bDebug:
            print('contours_gt')
            print(contours_gt)

        # Get contour area of GT
        if contours_gt:
            area = cv2.contourArea(np.array(contours_gt))
            areas_gt[target_class] = area

        # print("\tArea:", areas_gt[target_class])

        # Draw GT contours
        img = np.zeros_like(gt__)
        # print(img.shape)
        img[gt == target_class, 0] = 128  # Blue
        img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        pr = pr_.copy()
        pr[pr != target_class] = 0
        # print(pr.shape)

        # contours는 point의 list형태.
        if major == '3':  # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:  # For other opencv versions
            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # contours 는 list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())

        if bDebug:
            print('contours_pr')
            print(contours_pr)

        # Draw predicted contours
        img[pr == target_class, 2] = 128  # Red
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        # 3. calculate
        precision, numerator, denominator = calc_precision_recall(
            contours_gt, contours_pr, threshold)  # Precision
        # print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            contours_pr, contours_gt, threshold)  # Recall
        # print("\trecall:", denominator, numerator)

        f1 = 0
        try:
            f1 = 2 * recall * precision / (recall + precision)  # F1 score
        except:
            # f1 = 0
            f1 = np.nan
        # print("\tf1:", f1)
        bfscores[target_class] = f1

        # cv2.imshow('image', img)
        # cv2.waitKey(1000)

    # cv2.destroyAllWindows()

    # return bfscores[1:], np.sum(bfscores[1:])/len(classes[1:])    # Return bfscores, except for background, and per image score
    return bfscores[1:], areas_gt[1:]  # Return bfscores, except for background

if __name__ == "__main__":
    dict_score = {}
    for idx in tqdm(range(100)):

        sample_gt = f'/home/muyun99/data/dataset/Supervisely/train_mask/{idx+1}.png'
        sample_pred = f'/home/muyun99/github/detectron2/projects/PointRend/PointRend_mask/{idx+1}.png'

        score, areas_gt = bfscore(sample_gt, sample_pred, 2)
        total_area = np.nansum(areas_gt)

        fw_bfscore = []
        for each in zip(score, areas_gt):
            if math.isnan(each[0]) or math.isnan(each[1]):
                fw_bfscore.append(math.nan)
            else:
                fw_bfscore.append(each[0] * each[1])

        dict_score[sample_gt] = [np.nanmean(score)]
    df_score = pd.DataFrame.from_dict(dict_score, orient='index')
    df_score.to_csv("df_bfscore.csv")