from detectron2.engine import default_argument_parser
from liuy.adsampler.adsampler_dataloader import MyDataSet, transform
from liuy.adsampler.adsampler_model import Discriminator, VAE
from liuy.adsampler.util import read_img_list
import liuy.adsampler.config as config
from liuy.utils.local_config import coco_data
import torch
import torch.optim as optim
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import random
import os
import copy
from liuy.implementation.CoCoSegModel import CoCoSegModel
# from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from ranger import Ranger
from liuy.utils.local_config import VAE_feature_path, WEIGHT_path
import codecs
import csv
from PIL import Image
from torch.autograd import Variable

project_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
img_path = "/home/muyun99/Desktop/coco/train2014"

logger = logging.getLogger(__name__)
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="adsample.log")
logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.WARNING)
handler2.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)
logger.addHandler(handler1)
logger.addHandler(handler2)


def write_logger(log_str):
    print(log_str)
    logger.info(log_str)


class Adversary_sampler_trainer:
    def __init__(self, whole_image_id):
        """

        :param whole_image_id: list[int]
        """
        self.whole_image_id = whole_image_id
        self.coco_data = None
        self.all_imageid = None
        self.labeled_imageid = None
        self.unlabeled_imageid = None
        self.train_iterations = None

        self.image_size = config.IMAGE_SIZE
        self.z_dim = config.z_dim
        self.batch_size = config.BATCH_SIZE
        self.num_vae_steps = config.num_vae_steps
        self.num_dis_steps = config.num_dis_steps
        self.beta = config.beta
        self.adversary_param = config.adversary_param
        self.train_epochs = config.train_epochs

        self.vae = VAE(image_size=self.image_size, z_dim=self.z_dim)
        self.discriminator = Discriminator(z_dim=self.z_dim)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cuda = True
        self.device = torch.device('cuda')

    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD

    def read_data(self, dataloader):
        while True:
            for img, _, _ in dataloader:
                yield img, _, _

    def get_data_loader(self, all_imageid_path, labeled_imageid_path, coco_data):
        all_imageid = read_img_list(path=all_imageid_path)
        all_imageid = np.setdiff1d(list(all_imageid), list(error_imgid))
        if labeled_imageid_path is None:
            labeled_imageid = random.sample(list(all_imageid), int(len(all_imageid)*0.5))
        else:
            labeled_imageid = read_img_list(path=labeled_imageid_path)

        self.coco_data = coco_data
        self.all_imageid = all_imageid
        self.labeled_imageid = labeled_imageid
        self.unlabeled_imageid = np.setdiff1d(list(self.all_imageid), list(self.labeled_imageid))

        labeled_filename_list = []
        unlabeled_filename_list = []
        for num in self.labeled_imageid:
            labeled_filename_list.append(
                self.coco_data[0]['image_root'] + "/COCO_train2014_" + str(num).zfill(12) + ".jpg")
        for num in self.unlabeled_imageid:
            unlabeled_filename_list.append(
                self.coco_data[0]['image_root'] + "/COCO_train2014_" + str(num).zfill(12) + ".jpg")

        # 0代表labeled，1代表unlabeled
        temp_dict = {"filename": labeled_filename_list, "label": 0}
        labeled_df = pd.DataFrame(temp_dict)
        temp_dict = {"filename": unlabeled_filename_list, "label": 1}
        unlabeled_df = pd.DataFrame(temp_dict)

        labeled_data_set = MyDataSet(labeled_df, transform)
        labeled_data_loader = (DataLoader(labeled_data_set, batch_size=self.batch_size, shuffle=False, drop_last=False))

        unlabeled_data_set = MyDataSet(unlabeled_df, transform)
        unlabeled_data_loader = (
            DataLoader(unlabeled_data_set, batch_size=self.batch_size, shuffle=False, drop_last=False))

        return labeled_data_loader, unlabeled_data_loader

    def build_data_loader(self, coco_data):
        """
        the different between build_data_loader function and get_data_loader fun
        is the parameter

        :param labeled_imageid: list[int], if is None randomly select half data from
        self.whole_image_id as label_imageid
        :param coco_data:
        :return:
        """

        # turn the whole_image_id from list[int] to numpy array,
        # get unlabeled_imageid & labeled imageid from the whole_image_id
        all_imageid = np.array(self.whole_image_id)
        labeled_imageid = np.random.sample(all_imageid, int(all_imageid.size / 2), replace=False)
        unlabeled_imageid = np.setdiff1d(list(all_imageid), list(labeled_imageid))

        self.coco_data = coco_data
        self.all_imageid = all_imageid
        self.labeled_imageid = labeled_imageid
        self.unlabeled_imageid = unlabeled_imageid

        labeled_filename_list = []
        unlabeled_filename_list = []
        for num in self.labeled_imageid:
            labeled_filename_list.append(
                self.coco_data[0]['image_root'] + "/COCO_train2014_" + str(num).zfill(12) + ".jpg")
        for num in self.unlabeled_imageid:
            unlabeled_filename_list.append(
                self.coco_data[0]['image_root'] + "/COCO_train2014_" + str(num).zfill(12) + ".jpg")

        # 0代表labeled，1代表unlabeled
        temp_dict = {"filename": labeled_filename_list, "label": 0}
        labeled_df = pd.DataFrame(temp_dict)
        temp_dict = {"filename": unlabeled_filename_list, "label": 1}
        unlabeled_df = pd.DataFrame(temp_dict)

        labeled_data_set = MyDataSet(labeled_df, transform)
        labeled_data_loader = (DataLoader(labeled_data_set, batch_size=self.batch_size, shuffle=False, drop_last=False))

        unlabeled_data_set = MyDataSet(unlabeled_df, transform)
        unlabeled_data_loader = (
            DataLoader(unlabeled_data_set, batch_size=self.batch_size, shuffle=False, drop_last=False))

        return labeled_data_loader, unlabeled_data_loader

    def load_weight(self, vae_weight, dis_weight):
        self.vae.load_state_dict(torch.load(vae_weight))
        write_logger("--vae load weight success")

        self.discriminator.load_state_dict(torch.load(dis_weight))
        write_logger("--discriminator load weight success")

    def train_vae_dis(self, labeled_data_loader, unlabeled_data_loader):

        # Ranger激活函数库安装命令: pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git
        # optim_vae = Ranger(self.vae.parameters(), lr=1e-3, weight_decay=0.0005)
        # optim_discriminator = Ranger(self.discriminator.parameters(), lr=1e-3, weight_decay=0.0005)

        # 也可以使用Adam激活函数
        optim_vae = optim.Adam(self.vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=5e-4)

        # GradualWarmupScheduler安装命令: pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
        # 这里构造学习率的scheduler, 使用warmup技巧使初始学习率趋于稳定，之后再使用cosinelr scheduler
        scheduler_MultiStep_vae = MultiStepLR(optim_vae, milestones=[10, 20])
        scheduler_MultiStep_discriminator = MultiStepLR(optim_discriminator, milestones=[10, 20])
        # scheduler_Cosinelr_vae = CosineAnnealingLR(optim_vae, T_max=10)
        # scheduler_Cosinelr_discriminator = CosineAnnealingLR(optim_vae, T_max=10)
        # scheduler_warmup_vae = GradualWarmupScheduler(optim_vae, multiplier=1, total_epoch=5,
        #                                               after_scheduler=scheduler_Cosinelr_vae)
        # scheduler_warmup_discriminator = GradualWarmupScheduler(optim_discriminator, multiplier=1, total_epoch=5,
        #                                                         after_scheduler=scheduler_Cosinelr_discriminator)

        self.vae.train()
        self.discriminator.train()
        self.vae = self.vae.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.train_iterations = round(self.train_epochs * max(len(labeled_data_loader), len(unlabeled_data_loader)))
        save_name = len(labeled_data_loader) * self.batch_size
        iter_per_epoch = max(len(labeled_data_loader), len(unlabeled_data_loader))
        min_vae_loss = pow(10, 9)
        min_discriminator_loss = pow(10, 9)
        patient_vae_iter_count = 0
        patient_dis_iter_count = 0

        write_logger("--labeled batch:{}".format(len(labeled_data_loader)))
        write_logger("--unlabeled batch:{}".format(len(unlabeled_data_loader)))
        write_logger("--train_iteration:{}".format(self.train_iterations))

        labeled_data = self.read_data(labeled_data_loader)
        unlabeled_data = self.read_data(unlabeled_data_loader)
        start_time = time.time()
        for iter_count in range(self.train_iterations):
            if iter_count % iter_per_epoch == 0:
                optim_vae.step()
                optim_discriminator.step()
                scheduler_MultiStep_vae.step(int(iter_count / iter_per_epoch))
                scheduler_MultiStep_discriminator.step(int(iter_count / iter_per_epoch))

            iter_count += 1
            labeled_imgs, _, _ = next(labeled_data)
            unlabeled_imgs, _, _ = next(unlabeled_data)
            labeled_imgs = labeled_imgs.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)
            total_vae_loss = 0
            dsc_loss = 0

            # VAE step
            for count in range(self.num_vae_steps):
                recon, z, mu, logvar = self.vae(labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)

                unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs,
                                                  unlab_recon, unlab_mu, unlab_logvar, self.beta)

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                if self.cuda:
                    lab_real_preds = lab_real_preds.to(self.device)
                    unlab_real_preds = unlab_real_preds.to(self.device)

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_vae_steps - 1):
                    labeled_imgs, _, _ = next(labeled_data)
                    unlabeled_imgs, _, _ = next(unlabeled_data)
                    if self.cuda:
                        labeled_imgs = labeled_imgs.to(self.device)
                        unlabeled_imgs = unlabeled_imgs.to(self.device)

            # Discriminator step
            for count in range(self.num_dis_steps):
                with torch.no_grad():
                    _, _, mu, _ = self.vae(labeled_imgs)
                    _, _, unlab_mu, _ = self.vae(unlabeled_imgs)

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                if self.cuda:
                    lab_real_preds = lab_real_preds.to(self.device)
                    unlab_fake_preds = unlab_fake_preds.to(self.device)

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_dis_steps - 1):
                    labeled_imgs, _, _ = next(labeled_data)
                    unlabeled_imgs, _, _ = next(unlabeled_data)

                    if self.cuda:
                        labeled_imgs = labeled_imgs.to(self.device)
                        unlabeled_imgs = unlabeled_imgs.to(self.device)

            if iter_count % 10 == 0:
                end_time = time.time()
                if iter_count % 100 == 0:
                    write_logger("Current lr: {}".format(optim_vae.param_groups[0]['lr']))
                    write_logger("Current lr: {}".format(optim_discriminator.param_groups[0]['lr']))

                write_logger("--10 iters using {} s".format(end_time - start_time))
                write_logger('Current training iteration: {}'.format(iter_count))
                write_logger('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                write_logger('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))
                start_time = time.time()

            if iter_count % 2500 == 0:
                torch.save(self.vae.state_dict(),
                           os.path.join(WEIGHT_path, "vae_model_{}_{}.pth".format(save_name, iter_count, )))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(WEIGHT_path, "dis_model_{}_{}.pth".format(save_name, iter_count)))

            if total_vae_loss.item() < min_vae_loss:
                min_vae_loss = total_vae_loss.item()
                patient_vae_iter_count = 0
            else:
                patient_vae_iter_count += 1

            if dsc_loss.item() < min_discriminator_loss:
                min_discriminator_loss = dsc_loss.item()
                patient_dis_iter_count = 0
            else:
                patient_dis_iter_count += 1

            if patient_vae_iter_count >= config.patient_iter or patient_dis_iter_count >= config.patient_iter:
                write_logger("--over patient iter:{}, now iter: {}".format(config.patient_iter, iter_count))
                break

        project_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        torch.save(self.vae.state_dict(),
                   os.path.join(WEIGHT_path, "vae_model_{}_{}_{}.pth".format(save_name, project_start_time,
                                                                             project_end_time)))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(WEIGHT_path, "dis_model_{}_{}_{}.pth".format(save_name, project_start_time,
                                                                             project_end_time)))

    def get_csv(self, img_list_path, save_name):
        img_list = read_img_list(path=img_list_path)
        self.vae = self.vae.to(self.device)
        self.vae.eval()

        img_filename_list = []
        for img_id in img_list:
            img_filename_list.append(os.path.join(img_path, "COCO_train2014_" + str(img_id).zfill(12) + ".jpg"))

        features = []
        count = 0
        for single_img_filename in img_filename_list:
            img = Image.open(single_img_filename).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)

            count += 1

            recon, z, mu, logvar = self.vae(img)

            z = z.cpu().data
            for single_z in z:
                features.append(list(single_z.detach().numpy()))
            if count == 1000:
                print(count)
        write_logger("--compute feature done!")

        save_list = []
        length = len(features)
        for i in range(length):
            save_list.append([img_list[i], features[i]])
        name = ["image_path", "feature"]
        test_csv = pd.DataFrame(columns=name, data=save_list)
        test_csv.to_csv(save_name, encoding='utf-8')

    def save_vae_feature(self, save_name):
        """
        :param image_id: list[int] the id of the image which will be extracted feature,
        if image_id is None, whole image will be extracted feature
        :return:
        """
        img_list = self.whole_image_id

        self.vae = self.vae.to(self.device)
        self.vae.eval()

        img_filename_list = []
        for img_id in img_list:
            img_filename_list.append(
                os.path.join(coco_data[0]['image_root'], "COCO_train2014_" + str(img_id).zfill(12) + ".jpg"))
        features = []
        count = 0

        for single_img_filename in img_filename_list:
            img = Image.open(single_img_filename).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)

            count += 1
            recon, z, mu, logvar = self.vae(img)
            z = z.cpu().data
            for single_z in z:
                features.append(list(single_z.detach().numpy()))
            if count == 1000:
                print(count)
        write_logger("--compute feature done!")

        save_list = []
        length = len(features)
        for i in range(length):
            save_list.append([img_list[i], features[i]])
        name = ["image_path", "feature"]
        test_csv = pd.DataFrame(columns=name, data=save_list)
        test_csv.to_csv(save_name, encoding='utf-8')

    def sample(self, labeled_imageid_path, unlabeled_data_loader, n_sample, save_path):
        labeled_imageid = read_img_list(labeled_imageid_path)
        save_path = os.path.join(save_path, str(n_sample + len(labeled_imageid)))
        # labeled_imageid list[int]
        self.vae.eval()
        self.discriminator.eval()
        self.vae = self.vae.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        all_preds = []
        all_indices = []
        csv_list = []
        for images, _, indices in unlabeled_data_loader:
            images = images.to(self.device)
            with torch.no_grad():
                _, _, mu, _ = self.vae(images)
                preds = self.discriminator(mu)
            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)
            length = len(indices)
            for i in range(length):
                csv_list.append([indices[i], mu[i]])

        mu_csv = pd.DataFrame(columns=["index", "mu"], data=csv_list)
        # mu_csv.to_csv("mu.csv", encoding="utf-8")
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        print(n_sample)
        _, query_indices = torch.topk(all_preds, int(n_sample))
        query_pool_indices = np.asarray(all_indices)[query_indices]

        print(len(labeled_imageid))
        labeled_imageid.extend(list(query_pool_indices))
        print(query_pool_indices)

        file_csv = codecs.open(save_path, 'w+', 'utf-8')
        writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(query_pool_indices))
        print("save img_id_list successfully")


        print(len(labeled_imageid))
        labeled_imageid.extend(list(query_pool_indices))
        print(query_pool_indices)

        return query_pool_indices


# 用于检查所有图像能否可以被正确读取
# 结果：COCO_train2014_000000167126.jpg load error!
def check_file(img_list_path):
    list1 = []
    error_count = 0
    img_list = read_img_list(img_list_path)

    for single_img_id in img_list:
        single_img_path = config.coco_data[0]['image_root'] + "COCO_train2014_" + str(single_img_id).zfill(12) + ".jpg"
        try:
            Image.open(img_path).convert('RGB')
            list1.append(single_img_id)
        except:
            write_logger("--{} load error!".format(single_img_path))
            error_count += 1
    write_logger("--{} img load error!".format(error_count))
    return copy.deepcopy(list1)


def seed_torch(seed=61):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch()
    # 用于检查所有图像能否可以被正确读取
    # 结果：COCO_train2014_000000167126.jpg load error!
    # check_file("/home/muyun99/Downloads/Tsne/adsampler/imageid/all")

    # initialize seg_model and get the whole_image_id
    args = default_argument_parser().parse_args()
    seg_model = CoCoSegModel(args, project_id='adversely', coco_data=coco_data, resume_or_load=True)

    data_loader = seg_model.trainer.data_loader
    whole_image_id = [item['image_id'] for item in data_loader.dataset._dataset._lst]

    # 错误图像的id放在error_imgid
    error_imgid = [167126]
    trainer = Adversary_sampler_trainer(whole_image_id=whole_image_id)

    # 加载预训练模型
    trainer.load_weight(
        vae_weight=os.path.join(WEIGHT_path, "vae_model_14912_2500.pth"),
        dis_weight=os.path.join(WEIGHT_path, "dis_model_14912_2500.pth")
    )

    # 构造训练用的dataloader
    # labeled_data, unlabeled_data = trainer.get_data_loader(
    #     all_imageid_path="imageid/all",
    #     labeled_imageid_path="imageid/VAAL/2000",
    #     coco_data=coco_data
    # )
    labeled_data, unlabeled_data = trainer.build_data_loader(coco_data=coco_data)
    # 开始训练
    # trainer.train_vae_dis(labeled_data, unlabeled_data)

    # 传入的img_id生成隐变量的csv文件
    # trainer.get_csv2(
    #     img_list_path="/home/muyun99/Documents/al_ins_seg/liuy/adsampler/imageid/VAAL/2000",
    #     save_name=VAE_feature_path
    # )
    # trainer.get_csv1(
    #     img_list_path="/home/muyun99/Documents/al_ins_seg/liuy/adsampler/imageid/VAAL/2000",
    #     save_name=VAE_feature_path
    # )
    trainer.save_vae_feature(save_name=VAE_feature_path)

    # n_sample = 2000
    # trainer.sample(labeled_imageid_path="imageid/VAAL/2000",
    #                unlabeled_data_loader=unlabeled_data,
    #                n_sample=n_sample,
    #                save_path="imageid/VAAL/")
