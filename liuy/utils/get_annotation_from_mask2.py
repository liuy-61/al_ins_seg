import cv2
import numpy as np
import queue


def get_single_segmentation(image, pixel, x, y):
    single_segmentation = []
    width, height, channel = np.array(image).shape
    pixel_queue = queue.Queue()
    pixel_queue.put([x, y])

    already_check = np.zeros((width, height), dtype=np.int)

    add_x = [0, 0, 1, -1]
    add_y = [1, -1, 0, 0]
    while not pixel_queue.empty():
        now_pixel = pixel_queue.get()
        now_x = now_pixel[0]
        now_y = now_pixel[1]
        for i in range(4):
            next_x = now_x + add_x[i]
            next_y = now_y + add_y[i]
            if next_x < 0 or next_x >= width:
                continue
            if next_y < 0 or next_y >= height:
                continue

            next_pixel = []
            for j in range(channel):
                next_pixel.append(image[next_x][next_y][j])
            if already_check[next_x][next_y] == 0:
                # 如果下一个pixel的值等于传入的pixel的值时，将其加入队列中
                # 如果不是，则当前像素为当前mask的边界
                if next_pixel == pixel:
                    pixel_queue.put([next_x, next_y])
                    already_check[next_x][next_y] = 1
                else:
                    single_segmentation.append([next_x, next_y])
    return single_segmentation


def get_segmentation(image_path):
    segmentation = []
    BGR_image = cv2.imread(image_path)
    RGB_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)

    width, height, channel = RGB_image.shape
    image = list(RGB_image)

    # already_check_pixel表示已经检查过的颜色,[0,0,0]是背景，[224,224,192]是边界线
    already_check_pixel = [[0, 0, 0], [224, 224, 192]]

    for x in range(width):
        for y in range(height):
            pixel = []
            for i in range(channel):
                pixel.append(image[x][y][i])
            if pixel not in already_check_pixel:
                already_check_pixel.append(pixel)
                segmentation.append(get_single_segmentation(image, pixel, x, y))

    print("--compute done!")
    count = np.array(segmentation).shape[0]
    for i in range(count):
        segmentation[i] = np.array(segmentation[i])
    return segmentation


def polygon2bbox(polygons):
    """

    :param polygons: list[list[float]] ,each list[float] is one
    simple polygon in the format of [x1, y1, ...,xn,yn]

    :return: list[list[float]], each list[float] is one simple
    bounding box candidates, list[float]: [x_min, y_max, x_max-x_min, y_max-y_min]
    """
    bboxes = []
    for polygon in polygons:
        x = [item[0] for item in polygon]
        y = [item[1] for item in polygon]
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        # bboxes.append([x_min, y_max, x_max-x_min, y_max-y_min])
        bboxes.append([y_min, x_min, y_max, x_max])
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


if __name__ == '__main__':
    image_path = '/home/muyun99/Downloads/dataset/VOCdevkit/VOC2007/SegmentationObject/000170.png'
    # get_segmentation(image_path)
    seg = get_segmentation(image_path)

    bbox = polygon2bbox(seg)
    contours = bbox2contor(bbox)

    image = cv2.imread(image_path)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)