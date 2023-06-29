from torch.optim.lr_scheduler import _LRScheduler
import random
import numpy as np
import math
import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutout(inputs,targets):
    inputs_new = inputs.clone()
    input_s = torch.zeros_like(inputs)
    lam = np.random.rand()
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)

    inputs_new[:, :, bbx1:bbx2, bby1:bby2] = input_s[:, :, bbx1:bbx2, bby1:bby2]

    return inputs_new, targets


def mixup(inputs,targets, beta = 1.0):

    inputs_new = inputs.clone()
    N = inputs.size(0) # batch size
    indices = torch.randperm(N)
    inputs1 = inputs[indices, :, :, :].clone()
    targets1 = targets.clone()[indices]
    lam = np.random.beta(beta, beta)
    inputs_new = lam * inputs + (1 - lam) * inputs1

    return inputs_new, targets, targets1, lam

def mixup_criterion(criterion, pred, targets, targets1, lam):
    return lam * criterion(pred, targets) + (1 - lam) * criterion(pred, targets1)


def cutmix(inputs,targets, beta = 1.0):
    inputs_new = inputs.clone()
    N, _, H, W = inputs.shape # batch size

    indices = torch.randperm(N, device=torch.device('cuda'))
    # inputs1 = inputs[indices, :, :, :].clone()
    targets1 = targets.clone()[indices]
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    
    inputs_new[:, :, bbx1:bbx2, bby1:bby2] = inputs[indices, :, bbx1:bbx2, bby1:bby2].clone()
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    return inputs_new, targets, targets1, lam


def cutmix_criterion(criterion, pred, targets, targets1, lam):
    return lam * criterion(pred, targets) + (1 - lam) * criterion(pred, targets1)


def visulize():
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    import cv2
    
    cifar_data_path = '/remote-home/mfdu/nlpbeginner/hw2'
    cifar_train_transform = T.Compose([T.ToTensor(),])
    cifar_train = CIFAR100(cifar_data_path, download=True, transform=cifar_train_transform)
    cifar_train_loader = DataLoader(cifar_train, batch_size=8, shuffle=True)
    for i , data in enumerate(cifar_train_loader):
        if i == 320:
            break
    ori_imgs, _ = data
    cutout_imgs, _ = cutout(data)
    mixup_imgs, _, _, _ = mixup(data)
    cutmix_imgs, _, _, _ = cutmix(data)
    img_list = [ori_imgs, cutout_imgs, mixup_imgs, cutmix_imgs]
    from matplotlib import pyplot as plt
    index = 1
    fig, axes = plt.subplots(4, 3, figsize=(4, 4), tight_layout=True)
    for row in range(4):
        for col in range(3):
            axes[row, col].imshow(np.uint8(img_list[row][col,:,:,:].transpose(0,2)*255))
            #axes[row, col].axis('off')
            index += 1
    fig.savefig('save_img.jpg')
    plt.show()

