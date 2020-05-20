from __future__ import print_function
import json
from pycocotools.coco import COCO
# json_file='/media/tangyp/Data/coco/annotations/instances_val2014.json'
json_file='/media/tangyp/Data/coco/annotations/instances_train2014.json'
data=json.load(open(json_file,'r'))
coco = COCO(json_file)
data_2={}
annotation=[]
data_2['info']=data['info']
data_2['licenses']=data['licenses']
data_2['categories']=data['categories']
data_2['images'] = []
data_2['annotations'] = []
cnt = 0
# cat_ids = coco.getCatIds(catNms=['person'])
# img_ids = coco.getImgIds(catIds=cat_ids)
for img in data['images']:
    cnt += 1
    data_2['images'].append(img)
    img_id = img['id']
    ann_id = coco.getAnnIds(imgIds=img_id)
    ann = coco.loadAnns(ids=ann_id)
    data_2['annotations'].extend(ann)
    if cnt > 4000:
        break


# 保存到新的JSON文件，便于查看数据特点
json.dump(data_2,open('/media/tangyp/Data/coco/annotations/sub_train2014.json', 'w'), indent=4)
