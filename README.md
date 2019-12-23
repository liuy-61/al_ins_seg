detectron2的配置参考的detectron2
## 数据集结构 
数据集使用citysacpes

data_dir  /cityscapes
                         /leftImg8bit

                                     /train

                                     /val

                                     /test

                                     /sub_train

                         /gtfine

                                     /val

                                     /train

                                     /val

                                     /test

                                     /sub_train
                                     

data_dir  是自定义的路径  其他文件夹必须按照以上结构布置

ins_seg2.py 的使用

利用完整train 数据集 得到baseline
if __name__ == "__main__":
    data_dir = '######'   
    args = default_argument_parser().parse_args()
    
   （ 这里命令行输入配置文件参数 
    --config-file
    detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml）
    
    model = InsSegModel(args=args, project_id='train_on_complte_data', data_dir=data_dir)
    初始化模型
	
    model.fit()
    在train数据集上训练 得到baseline
	
     model.test()
    使用miou指标测试模型
    
	
在是用主动学算法 挑选出数据集 sub_train 之后 在sub_train上训练 并测试
  
   if __name__ == "__main__":
    data_dir = '######'   
    args = default_argument_parser().parse_args()
    
   （ 这里命令行输入配置文件参数 
    --config-file
    detectron2_origin/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml）
    
    model = InsSegModel(args=args, project_id='train_on_sub_data', data_dir=data_dir)
    初始化模型
	
    model.fit_on_subset()
    在train数据集上训练 得到baseline
	
     model.test()
    使用miou指标测试模型
    
    
   主动学习算法的接口 
   
   class BaseDataSlection(metaclass=ABCMeta):
    """Base data selection . The data selection object must inherit form this class."""

    def __init__(self, source_data_dir, target_data_dir, **kwargs):
        """
        :param source_data_dir: the root of the datasetto be selected
        :param target_data_dir: the root of the datasetto has been selected
        """
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir

    @abstractmethod
    def select(self, threshold, **kwargs):
        """select the data from source_data_dir and save to target_data_dir .

        """
  













