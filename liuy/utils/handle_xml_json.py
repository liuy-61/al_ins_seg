import xml.etree.ElementTree as ET
import json
import os

dir = '/media/tangyp/Data/coco/annotations'
file_name = 'sub_val2014.json'
json_file = os.path.join(dir, file_name)


def handle_xml(file_path, save_path=None):
    """

    :param file_path:
    :param save_path:  if save_path is None, means save_path is same as file_path.
    :return:
    """
    tree = ET.parse(file_path)
    # do something

def handle_json(file_path, save_path=None):
    """

        :param file_path:
        :param save_path:  if save_path is None, means save_path is same as file_path.
        :return:
    """
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    # do something

    with open(save_path, 'w') as f:
        json.dump(json_data, f)
