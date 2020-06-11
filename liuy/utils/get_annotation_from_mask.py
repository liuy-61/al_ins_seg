import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw


def get_segmentation(image_path):
    """

    :param image_path: Absolute path to the image
    like "/media/tangyp/Data/VOC/VOCdevkit/VOC2007/SegmentationObject/000170.png"
    :return: list[list[float]] ,each list[float] is one
    simple polygon in the format of [x1, y1, ...,xn,yn]
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        ls = []
        for item in contour:
            x = item[0][0]
            y = item[0][1]
            ls.append(x)
            ls.append(y)
        segmentation.append(ls)

    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imshow("img", image)
    cv2.waitKey(0)

    return segmentation


def polygon2bbox(polygons):
    """

    :param polygons: list[list[float]] ,each list[float] is one
    simple polygon in the format of [x1, y1, ...,xn,yn]

    :return: list[list[float]], each list[float] is one simple
    bounding box candidates, list[float]: [x_min, y_max, x_max-x_min, y_max-y_min]
    """
    bboxes =[]
    for polygon in polygons:
        x = [item for item in polygon if (polygon.index(item) % 2) == 0]
        y = [item for item in polygon if (polygon.index(item) % 2) != 0]
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        # bboxes.append([x_min, y_max, x_max-x_min, y_max-y_min])
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def bbox2contor(bboxes):
    """

    :param bboxes: list[list[float]], each list[float] is one simple
    bounding box candidates, list[float]: [x_min, y_min, x_max, y_max]
    :return:
    """
    contors = []

    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        contor = np.empty([4, 1, 2], dtype=int)
        contor[0, 0, 0] = x_min
        contor[0, 0, 1] = y_min

        contor[1, 0, 0] = x_max
        contor[1, 0, 1] = y_min

        contor[2, 0, 0] = x_max
        contor[2, 0, 1] = y_max

        contor[3, 0, 0] = x_min
        contor[3, 0, 1] = y_max

        contors.append(contor)
    return contors


def get_annotation(image_path):
    """

        :param image_path: Absolute path to the image
        like "/media/tangyp/Data/VOC/VOCdevkit/VOC2007/SegmentationObject/000170.png"

        :return: segmentations :list[list[float]] ,each list[float] is one
        simple polygon in the format of [x1, y1, ...,xn,yn]

                 bboxes : list[list[float]], each list[float] is one simple
        bounding box candidates, list[float]: [x_min, y_min, x_max, y_max]
        """

    segmentations = get_segmentation(image_path)
    bboxes = polygon2bbox(segmentations)
    return segmentations, bboxes

if __name__ == '__main__':
    image_path = '/media/tangyp/Data/VOC/VOCdevkit/VOC2007/SegmentationObject/000170.png'
    seg, bbox = get_annotation(image_path)
    """
    <class 'list'>: [[68, 374, 137, 374], [425, 155, 440, 191], [86, 97, 106, 165], [0, 87, 72, 223], [0, 20, 464, 374]]
    """
    contours = bbox2contor(bbox)

    # image = cv2.imread(image_path)
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)