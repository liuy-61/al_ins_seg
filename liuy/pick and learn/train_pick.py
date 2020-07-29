import os
import logging
from model import UNet_Pick
from utils.dataset_pick import BasicDataset, train_transform
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import torch
import argparse
import sys
from evaluate import eval_net_unet_pick
import torch_optimizer
from model import UNet
from utils.dice_loss_weight import dice_coeff
from utils.DiceLoss import BinaryDiceLoss

dir_img = "/home/muyun99/Desktop/supervisely/train/"
dir_mask = "/home/muyun99/Desktop/supervisely/train_mask/"
dir_checkpoint = '/media/muyun99/DownloadResource/dataset/opends-Supervisely Person Dataset/checkpoints'


class criterion(nn.Module):
    def __init__(self):
        super(criterion, self).__init__()

    def forward(self, true, pred, score):
        loss_fuc = nn.BCEWithLogitsLoss(weight=score.detach())
        return loss_fuc(pred, true)


def train_net(net,
              device,
              epochs=5,
              lr=0.01,
              batch_size=8,
              save_cp=True):
    net.to(device)
    train_dataset = BasicDataset(file_csv=args.train_csv,
                                 transform=train_transform)
    val_dataset = BasicDataset(file_csv=args.valid_csv,
                               transform=train_transform)
    test_dataset = BasicDataset(file_csv=args.test_csv,
                                transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                 drop_last=True)

    writer = SummaryWriter(comment="_{}".format(args.name))

    global_step = 0
    n_train = len(train_dataset)
    n_valid = len(val_dataset)

    logging.info(
        f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {lr}
                Training size:   {n_train}
                Validation size: {n_valid}
                Checkpoints:     {save_cp}
                Device:          {device}
                '''
    )

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    optimizer = torch_optimizer.Ranger(net.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_func = BinaryDiceLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_dataloader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.Unet.n_channels, \
                    'Network has been defined with {} input channels, '.format(
                        net.n_channels) + 'but loaded images have {} channels. Please check that '.format(
                        imgs.shape[1]) + 'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.Unet.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                mask_pred, weight_score = net(imgs, true_masks)
                loss = loss_func(predict=mask_pred, target=true_masks, score=weight_score)
                epoch_loss += loss

                writer.add_scalar('Loss/train', loss, global_step=global_step)
                # writer.add_scalar('score/train', weight_score.item(), global_step=global_step)

                pbar.set_postfix(**{'loss (batch)': loss})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

            # if global_step % (n_train // (10 * batch_size)) == 0:
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step=global_step)
                # writer.add_histogram('grads/' + tag, value.gard.data.cpu().numpy(), global_step=global_step)
            val_score = eval_net_unet_pick(net, val_dataloader, device)
            scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=global_step)

            if net.Unet.n_classes > 1:
                logging.info('Validation cross entropy: {}'.format(val_score))
                writer.add_scalar('Loss/valid', val_score, global_step=global_step)
            else:
                logging.info('Validation cross entropy: {}'.format(val_score))
                writer.add_scalar('Dice/valid', val_score, global_step=global_step)

            writer.add_images('images', imgs, global_step=global_step)
            if net.Unet.n_classes == 1:
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(mask_pred) > 0.5, global_step)

        if save_cp:
            dir_checkpoint = os.path.join("checkpoints", args.name)
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Create checkopint directory')
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'CP_epoch{}.pth'.format(epoch + 1)))
            logging.info('Checkpoint {} saved!'.format(epoch + 1))

    test_score = eval_net_unet_pick(net, test_dataloader, device)
    logging.info('Test Dice Coeff: {}'.format(test_score))
    writer.add_scalar('Dice/test', test_score, global_step=global_step)
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-train', '--train_csv', dest='train_csv', type=str, default=False,
                        help='train csv file_path')
    parser.add_argument('-valid', '--valid_csv', dest='valid_csv', type=str, default=False,
                        help='valid csv file_path')
    parser.add_argument('-test', '--test_csv', dest='test_csv', type=str, default=False,
                        help='test csv file_path')
    parser.add_argument('-n', '--name', dest='name', type=str, default="",
                        help='train name')

    return parser.parse_args()


if __name__ == '__main__':

    # for k, v in model.state_dict():
    #
    args = get_args()
    logging.basicConfig(filename=f'{args.name}.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_Pick(n_classes=1, n_channels=3)
    net.Unet.load_state_dict(
        torch.load("/home/muyun99/Desktop/MyGithub/person_seg/checkpoints/75_noise_pro/CP_epoch50.pth"))

    logging.info('Network:\n' +
                 '\t{} input channels\n'.format(net.Unet.n_channels) +
                 '\t{} output channels(classes)\n'.format(net.Unet.n_classes))
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info('Model loaded form {}'.format(args.load))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        train_net(net=net,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
