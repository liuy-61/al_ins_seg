import os
from liuy.utils.local_cofig import OUTPUT_DIR
import sys
import csv
import pickle
from liuy.implementation.CoresetSampler import cross_correlation
csv.field_size_limit(sys.maxsize)

def save_mask_feature( project_id, mask_feature, serial_number):
    """

    :param project_id:

    :param mask_feature: a list of dict, dict :{'image_id':int, 'feature_tensor':tensor}
    the tensors shape is (M,C,output_size,output_size)

    :param serial_number: Store the features part by part, ]
    and the serial_number Mark what part this is
    :return:
    """

    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id)
    with open(detail_output_dir + '/' +str(serial_number) + '.pkl', 'wb') as f:
        pickle.dump(mask_feature, f, pickle.HIGHEST_PROTOCOL)


def read_mask_feature(project_id, serial_number):
    detail_output_dir = os.path.join(OUTPUT_DIR, 'project_' + project_id)
    if os.path.exists(detail_output_dir + '/' +str(serial_number) + '.pkl'):
        with open(detail_output_dir + '/' +str(serial_number) + '.pkl', 'rb') as f:
            return pickle.load(f)



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
    a = read_mask_feature('coreset',0)
    # a = cross_correlation(mask_feature[0]['feature_tensor'], mask_feature[1]['feature_tensor'])
    b = read_mask_feature('coreset',3)
    debug = 1

