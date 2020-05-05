# Introduction
为方便同学们实现样本选择策略，我们设计了以下模块,
同学们不用关注于数据的加载，模型的训练和预测，评估等等细节,
只需要关注于如何实现样本选择策略。在按照给定的接口实现自定义的采样器之后,
利用`liuy/imlementation/CoCoAlmodel.py`文件就可以对自定义的采样器进行评估。
文档接下来的内容安排如下：

1. 文档会先给出一个调用`liuy/implementation/Almodel.py` 的例子<br>
2. 接下来将介绍采样器接口，以及在实现采样器接口的注意事项<br>
3. 最后介绍一些常用的模型使用接口，例如分割模型中计算损失和预测方法，以便于同学们实现自己的采样方法<br>

# Baseline
我们使用detectron 2实现的mask RCNN，在完整的coco与cityscapes数据集上进行训练（只保留'people'类），并将示例分割的结果转成了语义分割，然后计算了mIoU指标作为baseline：

- 在cityscapes数据集上测试的'miou': 0.36673763587127806
- 在coco数据集上测试的'miou': 0.651936

# Run a Demo 
我们可以运行`liuy/implementation/CoCoAlmodel.py` 文件，该实例中使用了随机采样器，分割模型在训练集中先抽取40%(seed_batch设为0.4)的数据进行训练，作为模型的初始化，随后利用随机采样器在训练集中每次抽取20%（batch_sise设为0.2）的数据样本，直到样本全都选择完，
在采样器每次采样之后，分割模型再利用采样数据进行训练，并进行评估（评估指标为miou）,记录下每次评估结果<br>
在运行实例之前，首先需要修改配置，输入数据集路径与输出路径等。

##  Path Configuration
1. 找到`liuy/implementation/CoCoAlmodel.py`中的以下代码：
```
if __name__ == "__main__":
    coco_data = [{'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                  'image_root': '/media/tangyp/Data/coco/train2014'
                  },
                 {
                     'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                     # 'json_file': '/home/tangyp/liuy/detectron2_origin/liuy/utils/sub_val2014.json',
                     'image_root': '/media/tangyp/Data/coco/val2014'
                 }]
```
以上四条路径分别为训练集的标记json文件，image路径，测试集的标记json文件，及其image路径。请将其修改为本地数据数据集的路径。

2. 修改`liuy/utils/torch_utils.py`文件
```
OUTPUT_DIR = '/home/tangyp/liuy/mode_file/OUTPUT'
```
OUTPUT_DIR为模型中间文件和训练后模型的保存路径，按实际情况修改这个路径。

4. 对于使用了IDE的同学，请在IDE中设置以下命令行参数。如果直接使用命令行，请在运行时直接带上以下参数
```
--config-file
detectron2_origin/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
```
至此可以运行`liuy/implementation/CoCoAlmodel.py`文件

## Code Analyses
我们将代码流程总结如下：
```
seg_model = CoCoSegModel(args, project_id='coco', coco_data=coco_data, resume_or_load=False)
```
初始化分割模型，project_id 可自定义，它和文件输出路径有关。resume_or_load标志每次挑选完数据之后是否加载上次训练的权重。<br>
```
data_loader = seg_model.trainer.data_loader
```
得到分割模型的data_loader, 此时的data_loder的数据是整个训练集。<br>
```
randomsampler = CoCoRandomSampler('randomsampler', data_loader)
```
初始化随机采样器，'randomsampler' 为采样器的名字，在输出采样器评估结果时会用到。<br>
```
generate_one_curve(coco_data=coco_data,
                       data_loader=data_loader,
                       sampler=randomsampler,
                       ins_seg_model=seg_model,
                       batch_size=0.2,
                       seed_batch=0.4
                       )

```
先随机从训练集抽取百分之四十的样本（ seed_batch=0.4）作为训练样本，用于实例分割模型（seg_model）的初始训练，之后利用采样器randomsampler每一次从训练集中抽取百分之二十的训练样本（ batch_size=0.2），将样本加入训练样本之后，分割模型用新的训练样本进行训练，再对本轮训练好的分割模型进行评估，并保存评估结果。直到训练集中所有的样本都被采样完。有任意一次评估结果优于baseline则说明采样器有效。seed_batch和batch_size参数可以调动。

# Interface 

## BaseSampler :the interface of a sampler 
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
            self.image_files_list.append(item['image_id'])

    def select_batch(self, n_sample, already_selected, **kwargs):
        """
        file_name as key to data
        :param n_sample: batch size
        :param already_selected: list of image_id already selected
        :param kwargs:
        :return: list of image_id you selected this batch
        """
        return

```

## RandomSampler :examples of implementing interface
```
class CoCoRandomSampler(BaseSampler):
    def __init__(self, sampler_name, data_loader):
        super(CoCoRandomSampler, self).__init__(sampler_name, data_loader)
        self.image_files_list = []
        lt = data_loader.dataset._dataset._lst
        # file_name as key to data
        for item in lt:
            self.image_files_list.append(item['image_id'])

    def select_batch(self, n_sample, already_selected):
        cnt = 0
        samples = []
        while cnt < n_sample:
            sample = random.sample(self.image_files_list, 1)
            if sample[0] not in already_selected and sample[0] not in samples:
                cnt += 1
                samples.append(sample[0])

        assert len(samples) == n_sample
        assert len(set(samples)) == len(samples)
        return samples

```

`select_batch`函数的参数含义:<br>
`n_sample`为每个batch 选择的样本个数<br>
`already_selected` 为之前已经选择过的样本，already_selected也是一个`list`,可以将already_selected看作`self.image_files_list`的子集<br>
`select_batch`函数挑选样本时,应该在`self.image_files_list`挑选出与`already_selected`互斥的一个子集，并返回它。<br>

# Custom your own sampler in AlModel.py

实现`liuy/Interface/BaseSampler`自定义采样器之后，替换掉其中的随机采样器<br>

```
randomsampler = RandomSampler('randomsampler', data_loader) 
```
修改为
```
customsampler = CustomSampler('customsampler', data_loader) 
``` 
CustomSampler 为自定义的采样器<br>
然后运行CoCoAlModel.py

# Tool Functions
## compute_loss()
在`liuy/implementation/CoCoSegModel.py`中

```
if __name__ == "__main__":
    coco_data = [{'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                 'image_root': '/media/tangyp/Data/coco/train2014'
                 },
                 {
                  # 'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                  'json_file': '/home/tangyp/liuy/detectron2_origin/liuy/utils/sub_val2014.json',
                  'image_root': '/media/tangyp/Data/coco/val2014'
                 }]

    args = default_argument_parser().parse_args()
    model = CoCoSegModel(args, project_id='coco_test', coco_data=coco_data, resume_or_load=False
                         )
    # model.fit()
    # model.test()
    # model.predict()
    loss = model.compute_loss(coco_data[1]['json_file'], coco_data[1]['image_root'])
```
提供了`def compute_loss(self, json_file, image_root):` 方法，其中参数定义如下：

1. `image_root`为预测的image的路径
2. `json_file`为对应的label路径

注意，运行时需要使用命令行参数指定分割模型的配置文件：

```
--config-file
detectron2_origin/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml
```
该函数的返回值为一个list，其中的元素形如：
```
<class 'dict'>:
{'loss_cls': tensor(102.9732, device='cuda:0'),
'loss_box_reg': tensor(130.7693, device='cuda:0'),
'loss_mask': tensor(11.4862, device='cuda:0'),
'loss_rpn_cls': tensor(59.3035, device='cuda:0'),
'loss_rpn_loc': tensor(1.9601, device='cuda:0'),
'file_name': '/media/tangyp/Data/cityscape/leftImg8bit/sub_train/aachen/aachen_000002_000019_leftImg8bit.png'}
```


## predict_proba()
在`liuy/implementation/InsSegModel.py`中
```
if __name__ == "__main__":
    coco_data = [{'json_file': '/media/tangyp/Data/coco/annotations/instances_train2014.json',
                 'image_root': '/media/tangyp/Data/coco/train2014'
                 },
                 {
                  # 'json_file': '/media/tangyp/Data/coco/annotations/instances_val2014.json',
                  'json_file': '/home/tangyp/liuy/detectron2_origin/liuy/utils/sub_val2014.json',
                  'image_root': '/media/tangyp/Data/coco/val2014'
                 }]

    args = default_argument_parser().parse_args()
    model = CoCoSegModel(args, project_id='coco', coco_data=coco_data, resume_or_load=False
                         )
    prediction = model.predict_proba(coco_data[1]['json_file'], coco_data[1]['image_root'])
```

提供了`def predict_proba(self, json_file, image_root, conf_thres=0.7, nms_thres=0.4,verbose=True, **kwargs):`函数，用于预测样本。这里假设模型已经训练好，若没有，则需调用`model.fit()`来训练模型。
                      
参数说明同`compute_loss()`。类似的，该函数返回值为一个list，list元素为字典，字典元素如下所示:

```
{'file_name': '/media/tangyp/Data/cityscape/leftImg8bit/sub_train/aachen/aachen_000000_000019_leftImg8bit.png',
'boxes': Boxes(tensor([[1830.1968,  433.6077, 1889.0730,  548.3942],
        [ 890.5734,  446.4362,  912.8923,  498.7133],
        [ 914.9539,  440.7620,  938.4851,  496.9281]], device='cuda:0')),
        'labels': tensor([0, 0, 0], device='cuda:0'),
        'scores': tensor([0.9377, 0.8424, 0.7417],device='cuda:0')}
        'masks':  tensor([[[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],
        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False]],
        [[False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         [False, False, False,  ..., False, False, False],
         ...,                                              <br>
         [False, False, False,  ..., False, False, False], <br> 
         [False, False, False,  ..., False, False, False], <br>
         [False, False, False,  ..., False, False, False]]], device='cuda:0')
```       

# Bug Report

如果在使用本框架时有任何问题，以及需要其他工具函数，可以直接提issue或微信讨论。


