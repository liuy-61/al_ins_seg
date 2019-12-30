# 前言
为方便同学们实现样本选择策略，我们设计了以下模块<br>
同学们不用关注于数据的加载，模型的训练和预测，评估等等细节<br>
只需要关注于如何实现样本选择策略。在按照给定的接口实现自定义的采样器之后 <br>
利用liuy /imlementation/ Almodel.py文件就可以对自定义的采样器进行评估<br>
以下文档会先介绍如何运行一个实例，liuy/implementation/Almodel.py <br>
再介绍采样器接口，以及在实现采样器接口需要注意的细节<br>
然后介绍了提供的方法（在实现样本选择策略的时候或许要用到），分割模型中计算损失和预测方法<br>
  
# 实例运行
文件中我们可以运行liuy/implementation/Almodel.py 文件，该实例中使用了随机采样器，分割模型在训练集中先抽取20%(seed_batch设为0.2)的数据进行训练，作为模型的初始化<br>
随后利用随机采样器在训练集中每次抽取20%（batch_sise设为0.2）的数据样本，直到样本全都选择完<br>
在采样器每次采样之后，分割模型再利用采样数据进行训练，并进行评估（评估指标为miou）,记录下每次评估结果<br>
在运行实例之前，首先需要配置cityscapes数据集<br>

## 数据集配置路径
>cityscapes
>>leftImg8bit
>>>train<br>
>>>val<br>
>>>test<br>

>>gtfine
>>>train<br>
>>>val<br>
>>>test<br>

数据集的路径需按以上格式配置，如训练集的image_dir 为**/cityscapes/leftImg8bit/train  gt_dir 为**/cityscapes/gtfine/train

## 按实际情况修改liuy/implementation/Almodel.py 中的变量或参数
源码为 ：<br>
```
if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    seg_model = InsSegModel(args=args, project_id='AlModel', data_dir=data_dir)
    data_loader = seg_model.trainer.data_loader
    randomsampler = RandomSampler('randomsampler', data_loader)
    generate_one_curve(image_dir=image_dir,
                       gt_dir=gt_dir,
                       data_loader=data_loader,
                       sampler=randomsampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.2
                       )
```
1、image_dir、gt_dir分别修改为自己训练集的图像、标签路径。<br>
2、data_dir修改为自己 cityscapes的父目录，可参考例子中代码理解。<br>
3、设置命令行参数 设置分割模型的配置文件。<br>
```
--config-file
detectron2_origin/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
```
以上代码流程为：<br>
```
seg_model = InsSegModel(args=args, project_id='AlModel', data_dir=data_dir)
```
初始化分割模型  project_id 可自定义，它和文件输出路径有关。<br>
```
data_loader = seg_model.trainer.data_loader
```
得到分割模型的data_loader, 此时的data_loder的数据是整个训练集。<br>
```
randomsampler = RandomSampler('randomsampler', data_loader)
```
初始化随机采样器，'randomsampler' 为采样器的名字，在输出采样器评估结果时会用到。<br>
```
generate_one_curve(    image_dir=image_dir,
                       gt_dir=gt_dir,
                       data_loader=data_loader,
                       sampler=randomsampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.2
                       )
```
该函数 ： 先随机从训练集抽取百分之二十的样本（ seed_size=0.2）作为训练样本，用于实例分割模型（seg_model）的初始训练，之后利用采样器randomsampler每一次从训练集中抽取百分之二十的训练样本（ batch_size=0.2），在每一次采样器采取到一个bactch_size的样本，将样本加入训练样本之后，分割模型用新的训练样本进行训练，再对本轮训练好的分割模型进行评估，并保存评估结果。直到训练集中所有的样本都被采样完。有任意一次评估结果优于baseline则说明采样器有效。seed_size和batch_size参数可以调动。

# 如何实现采样器接口

我们最重要的就是实现样本选择策略即实现采样器接口<br>
实现liuy /Interface/ BaseSampler自定义采样器之后，替换掉liuy /implementation/ Almodel.py中的随机采样器<br>

```
randomsampler = RandomSampler('randomsampler', data_loader) 
```
修改为
```
customsampler = CustomSampler('customsampler', data_loader) 
``` 
CustomSampler 为自定义的采样器<br>
## 采样器接口  
```
class BaseSampler(metaclass=ABCMeta):
    def __init__(self, sampler_name, data_loader, **kwargs):
        """

        :param data_loader: we use the data_loader to init a  image_files_list, then we select data from image_files_list.
        :param sampler_name
        :param kwargs:
        """
        self.data_loader = data_loader
        self.sample_name = sampler_name
        self.image_files_list = []
        lt = data_loader.dataset._dataset._lst
        # file_name as key to data
        for item in lt:
            self.image_files_list.append(item['file_name'])

    def select_batch(self, n_sample, already_selected, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param already_selected: list of file_name already selected
        :param kwargs:
        :return: list of file_name you selected this batch
        """
        return
```

BaseSampler在传入sampler_name、data_loader参数初始化之后，会得到 self.image_files_list，<br>
self.image_files_list中的元素'file_name'为cityscapes数据集中一张图像数据的唯一标志，是一张图片的全路径。<br>
在初始化BaseSampler后，self.image_files_list。包含了训练集中所有的图像数据。<br>
BaseSampler接口的实现可以可以参考RandomSampler。<br>

## 随机采样器
```
import random

from liuy.Interface.BaseSampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(RandomSampler, self).__init__(sampler_name, data_loader)


    def select_batch(self, n_sample, already_selected):
        cnt = 0
        samples = []
        while cnt < n_sample:
           sample = random.sample(already_selected, 1)
           if sample not in already_selected:
               cnt += 1
               samples.append(sample)
        assert len(samples) == n_sample
        return samples

```
初始化函数没有扩展<br>
select_batch函数的参数含义:<br>
n_sample为每个batch 选择的样本个数<br>
already_selected 为之前已经选择过的样本，already_selected也是一个list,可以将already_selected看作self.image_files_list的子集<br>
select_batch函数挑选样本时,应该在self.image_files_list挑选出与already_selected互斥的一个子集，并返回它。<br>


# 代码目录（主要用到的）
>detectron2_origin
>>liuy
>>>implementation<br>
>>>interface<br>
>>>utils

# 提供的方法
## 计算损失
在liuy/implementation/InsSegModel.py中<br>
```
if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/sub_train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/sub_train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    model = InsSegModel(args=args, project_id='1', data_dir=data_dir)
    model.fit()
    losses = model.compute_loss(image_dir=image_dir,gt_dir=gt_dir)
    for loss in losses:
    print(loss)
```
在实例分割模型中 提供了def compute_loss(self, image_dir, gt_dir): 方法<br>
运行以上liuy/implementation/InsSegModel.py文件<br>
1、data_dir 为cityscapes数据集的父级目录<br>
2、image_dir 为计算的image的路径   gt_dir为对应的label路径<br>
3、命令行设置分割模型的配置文件
```
--config-file
configs/Cityscapes/mask_rcnn_R_50_FPN.yaml
```
4、如果分割模型之前已经训练完毕 则model.fit()可以注释掉<br>
运行以上代码，调用compute_loss后<br>
返回值为一个list，list元素为字典,字典元素如下所示<br>
<class 'dict'>: <br>
{'loss_cls': tensor(102.9732, device='cuda:0'), <br>
'loss_box_reg': tensor(130.7693, device='cuda:0'),<br>
'loss_mask': tensor(11.4862, device='cuda:0'), <br>
'loss_rpn_cls': tensor(59.3035, device='cuda:0'), <br>
'loss_rpn_loc': tensor(1.9601, device='cuda:0'), <br>
'file_name': '/media/tangyp/Data/cityscape/leftImg8bit/sub_train/aachen/aachen_000002_000019_leftImg8bit.png'}<br>


## 模型预测
在liuy/implementation/InsSegModel.py中<br>
```
if __name__ == "__main__":
    image_dir = '/media/tangyp/Data/cityscape/leftImg8bit/sub_train'
    gt_dir = '/media/tangyp/Data/cityscape/gtFine/sub_train'
    data_dir = '/media/tangyp/Data'
    args = default_argument_parser().parse_args()
    model = InsSegModel(args=args, project_id='1', data_dir=data_dir)
    model.fit()
    probability = model.predict_proba(image_dir, gt_dir)
```

在实例分割模型中 提供了def predict_proba(self, image_dir, gt_dir, conf_thres=0.7, nms_thres=0.4,
                      verbose=True, **kwargs):<br>
                      
运行以上liuy/implementation/InsSegModel.py文件<br>
1、data_dir 为cityscapes数据集的父级目录<br>
2、image_dir 为预测的image的路径   gt_dir为对应的label路径<br>
3、命令行设置分割模型的配置文件<br>
```
--config-file
configs/Cityscapes/mask_rcnn_R_50_FPN.yaml
```
4、如果分割模型之前已经训练完毕 则model.fit()可以注释掉<br>
按以上要求运行该文件，调用predict_proba之后，
返回值为一个list，list元素为字典，字典元素如下所示<br>

{'file_name': '/media/tangyp/Data/cityscape/leftImg8bit/sub_train/aachen/aachen_000000_000019_leftImg8bit.png',<br>
'boxes': Boxes(tensor([[1830.1968,  433.6077, 1889.0730,  548.3942],<br>
        [ 890.5734,  446.4362,  912.8923,  498.7133],<br>
        [ 914.9539,  440.7620,  938.4851,  496.9281]], device='cuda:0')), <br>
        'labels': tensor([0, 0, 0], device='cuda:0'), <br>
        'scores': tensor([0.9377, 0.8424, 0.7417],device='cuda:0')} <br>
       




