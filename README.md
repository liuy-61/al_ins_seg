# 数据集结构
数据集使用cityscapes
>cityscapes
>>leftImg8bit
>>>train<br>
>>>val<br>
>>>test<br>

>>gtfine
>>>train<br>
>>>val<br>
>>>test

文件夹必须按照以上结构布置

# 代码目录
>detectron2_origin
>>liuy
>>>implementation<br>
>>>interface<br>
>>>utils

主动学习的框架在impletation中

# 代码运行
运行implementation中的 ALModel.py
## 命令行参数
--config-file<br>
detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml
## if __name__ == "__main__": 下的变量修改
image_dir和gt_dir按实际情况修改 <br>
data_dir修改为cityscapes的父级目录<br>
## 运行流程

# 接口实现
主要实现在interface中提供的BaseSampler接口<br>
impeachmention中提供了RandomSampler作为参考<br>
实现自定义的sampler之后 在运行ALModel.py中替换RandomSampler 评估自定义的sampler<br>

# 提供的方法
## 计算损失
在liuy/implementation/InsSegModel.py中<br>
在实例分割模型中 提供了def compute_loss(self, image_dir, gt_dir): 方法<br>
image_dir 为计算的image的路径<br>
gt_dir     为对应的label路径<br>
返回值为一个list，元素为一个字典，字典包含了图像的file_name 和loss_cls，loss_box_reg，loss_mask，loss_rpn_cls，loss_rpn_loc

## 模型预测
在liuy/implementation/InsSegModel.py中<br>
在实例分割模型中提供了<br>
def predict_proba(self, image_dir, gt_dir, conf_thres=0.7, nms_thres=0.4,
                      verbose=True, **kwargs):<br>
                      
返回值为一个list list元素为一个字典，字典包含了预测图像的file_name，boxes坐标，instances类别，instances分数
       




