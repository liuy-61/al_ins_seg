import csv
import codecs
import pandas as pd
import os
from liuy.utils.torch_utils import OUTPUT_DIR
import sys
import csv
csv.field_size_limit(sys.maxsize)
def save_img_list( project_id, iteration, img_id_list):
    """
    :param dir:
    :param img_id_list:
    :return:
    """
    dir = OUTPUT_DIR + '/' + 'selected_img_list' + '/' + project_id
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = dir + '/' + str(iteration)
    file_csv = codecs.open(file, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(img_id_list)
    print("save img_id_list successfully")


def read_img_list( project_id, iteration):
    dir = OUTPUT_DIR + '/' + 'selected_img_list' + '/' + project_id
    if not os.path.exists(dir):
        os.makedirs(dir)
    file = dir + '/' + str(iteration)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        img_str = result[0][0]
        img_str = img_str.split(' ')
        img_list = [int(item) for item in img_str]
        debug = 1
        return img_list

def get_iter(project_id):
    """
    See how many iter should be by checking to see if the file exists
    :return:
    """
    iter = 0
    dir = OUTPUT_DIR + '/' + 'selected_img_list' + '/' + project_id

    while True:
        file = dir + '/' + str(iter)
        if os.path.exists(file) == True:
            iter += 1
        else:
            break
    return iter

if __name__ == '__main__':
    img_id_list = [5, 4, 6, 1, 5, 2, 7, 7, 233]
    project_id = 'coco'
    save_img_list(project_id=project_id, iteration=2, img_id_list=img_id_list)
    read_img_list(project_id=project_id, iteration=2)