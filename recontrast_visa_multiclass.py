# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.recontrast import ReContrast
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, visualize, global_cosine_hm, global_cosine, replace_layers
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
import collections
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list):
    setup_seed(args.seed)

    total_iters = 5000
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = '../VisA_pytorch/1cls/' + item + '/train'
        test_path = '../VisA_pytorch/1cls/' + item

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
                            for test_data in test_data_list]

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    replace_layers(decoder, nn.ReLU, nn.GELU())

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    optimizer = torch.optim.AdamW([{'params': decoder.parameters()}, {'params': bn.parameters()},
                                   {'params': encoder.parameters(), 'lr': 1e-5}],
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_iters * 0.8)], gamma=0.2)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train(encoder_bn_train=False)

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)

            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            lr_scheduler.step()

            if (it + 1) % 1000 == 0:
                auroc_px_list, auroc_sp_list, ap_px_list, ap_sp_list = [], [], [], []

                for item, test_dataloader in zip(item_list, test_dataloader_list):
                    auroc_px, auroc_sp, ap_px, ap_sp = evaluation_batch(model, test_dataloader, device,
                                                                        reg_calib=False,
                                                                        max_ratio=0.01)

                    auroc_px_list.append(auroc_px)
                    auroc_sp_list.append(auroc_sp)
                    ap_px_list.append(ap_px)
                    ap_sp_list.append(ap_sp)

                    print_fn('{}: P-Auroc:{:.4f}, I-Auroc:{:.4f},P-AP:{:.4f},I-AP:{:.4f}'.format(item,
                                                                                                 auroc_px,
                                                                                                 auroc_sp,
                                                                                                 ap_px, ap_sp))
                print_fn('Mean: P-Auroc:{:.4f}, I-Auroc:{:.4f},P-AP:{:.4f},I-AP:{:.4f}'.format(np.mean(auroc_px_list),
                                                                                               np.mean(auroc_sp_list),
                                                                                               np.mean(ap_px_list),
                                                                                               np.mean(ap_sp_list)))

                model.train(encoder_bn_train=False)

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_mvtec_unify_max1_it5k_lr2e31e52e4_slr01_b16_s1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    args = parser.parse_args()

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
