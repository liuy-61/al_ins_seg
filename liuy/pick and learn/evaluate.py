import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_loss_weight import dice_coeff


def eval_net_unet(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in tqdm(loader):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            total_loss += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            total_loss += dice_coeff(pred, true_masks).item()

    net.train()
    return total_loss / n_val

def eval_net_unet_pick(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in tqdm(loader):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred, weight_score = net(imgs, true_masks)

        if net.n_classes > 1:
            total_loss += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            total_loss += dice_coeff(pred, true_masks).item()
    net.train()
    return total_loss / n_val
