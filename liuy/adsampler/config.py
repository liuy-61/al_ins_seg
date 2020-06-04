# 训练的epoch数
train_epochs = 50
IMAGE_SIZE = 128
BATCH_SIZE = 64
coco_data = [{
    'json_file': '/home/muyun99/Desktop/coco/annotations/instances_train2014.json',
    'image_root': '/home/muyun99/Desktop/coco/train2014/'
}]

# 隐变量的维度
z_dim = 64

# 每一步中训练vae和dis的次数
num_vae_steps = 1
num_dis_steps = 1

# 损失函数的超参数
beta = 1.0
adversary_param = 10.0

# 经过这么多轮loss不再降低即停止程序
patient_iter = 100
