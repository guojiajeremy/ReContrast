
import torch
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2
from models.recontrast import ReContrast, ReContrast
from dataset import MedicalDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_noseg, visualize_noseg
from utils import global_cosine, global_cosine_hm

from torch.nn import functional as F
from functools import partial

import warnings
import copy
import logging

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


def train(_class_):
    print_fn(_class_)
    setup_seed(111)

    total_iters = 1000
    batch_size = 32
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '../APTOS/'
    test_path = '../APTOS/'

    train_data = MedicalDataset(root=train_path, transform=data_transform, phase="train")
    test_data = MedicalDataset(root=test_path, transform=data_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))

    auroc_sp_best = 0
    it = 0
    for epoch in range(total_iters // len(train_dataloader) + 1):
        model.train(encoder_bn_train=True)
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            # loss = global_cosine(en[:3], de[:3]) / 2 + \
            #        global_cosine(en[3:], de[3:]) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            if (it + 1) % 250 == 0:
                auroc, f1, acc = evaluation_noseg(model, test_dataloader, device)
                model.train(encoder_bn_train=True)
                print_fn('AUROC:{:.4f}, F1:{:.4f}, ACC:{:.4f}'.format(auroc, f1, acc))
                if auroc >= auroc_sp_best:
                    auroc_sp_best = auroc
            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    visualize_noseg(model, test_dataloader, device, _class_=_class_)
    return auroc, auroc_sp_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_aptos_b32_it1k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    item_list = ['APTOS']
    for item in item_list:
        train(item)
