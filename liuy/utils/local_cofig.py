
# debug_data is the subset of coco_data
coco_data = [{  'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                'image_root': '/media/tangyp/Data/coco/train2014'
             },
             {
                'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                # 'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',
                'image_root': '/media/tangyp/Data/coco/val2014'
              },
            ]

debug_data = [{ 'json_file': '/media/tangyp/Data/coco/annotations/sub_train2014.json',
                'image_root': '/media/tangyp/Data/coco/train2014'
             },
             {
                'json_file': '/media/tangyp/Data/coco/annotations/sub_val2014.json',
                'image_root': '/media/tangyp/Data/coco/val2014'
              },
            ]

OUTPUT_DIR = '/media/tangyp/Data/model_file/OUTPUT_DIR'

MODEL_NAME = {
    'Faster_RCNN': '/home/tangyp/liuy/detectron2_origin/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',

    'Mask_RCNN': '/home/tangyp/liuy/detectron2_origin/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',

    'Mask_RCNN2': '/home/tangyp/liuy/al_ins_seg/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',

    }