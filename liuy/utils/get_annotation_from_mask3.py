import cv2
import numpy as np
import queue


def get_single_segmentation(image, pexel, x, y):
    single_segmentation = []
    width, height, channel = np.array(image).shape
    pexel_queue = queue.Queue()
    pexel_queue.put([x, y])

    already_check = np.zeros((width, height), dtype=np.int)

    # offset_x 和 offset_y 是为了遍历当前像素点的上下左右的像素点
    offset_x = [0, 0, 1, -1]
    offset_y = [1, -1, 0, 0]
    while not pexel_queue.empty():
        now_pexel = pexel_queue.get()
        now_x = now_pexel[0]
        now_y = now_pexel[1]
        for i in range(4):
            next_x = now_x + offset_x[i]
            next_y = now_y + offset_y[i]
            if next_x < 0 or next_x >= width:
                continue
            if next_y < 0 or next_y >= height:
                continue

            next_pexel = []
            for j in range(channel):
                next_pexel.append(image[next_x][next_y][j])
            if already_check[next_x][next_y] == 0:
                # 如果下一个pexel的值等于传入的pexel的值时，将其加入队列中
                # 如果不是，则当前像素为当前mask的边界
                if next_pexel == pexel:
                    pexel_queue.put([next_x, next_y])
                    already_check[next_x][next_y] = 1
                else:
                    single_segmentation.append([next_x, next_y])
    return single_segmentation


def get_segmentation(image_path):
    segmentation = []
    BGR_image = cv2.imread(image_path)
    RGB_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB)
    height, width, channel = RGB_image.shape

    # 由于opencv的img读进来是宽x高x通道数的数组，我们将其转成高x宽x通道数的数组，转置一下即可
    image = list(RGB_image.transpose((1, 0, 2)))

    # already_check_pexel表示已经检查过的颜色,[0,0,0]是背景，[224,224,192]是边界线
    already_check_pexel = [[0, 0, 0], [224, 224, 192]]

    for x in range(width):
        for y in range(height):
            pexel = []
            for i in range(channel):
                pexel.append(image[x][y][i])

            # 如果当前遍历的像素值尚未访问到，说明这是一个新的分割掩膜(用新的颜色标注)
            if pexel not in already_check_pexel:
                already_check_pexel.append(pexel)
                segmentation.append(
                    get_single_segmentation(image, pexel, x, y))

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


if __name__ == '__main__':
    image_path = "./123.png"

    seg = get_segmentation(image_path)
    bbox = polygon2bbox(seg)
    contours = bbox2contor(bbox)

    image = cv2.imread(image_path)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
