import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial

from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1))) * weight[item]
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1))) * weight[item]
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1),
                                        b_.view(b_.shape[0], -1))) * weight[item]
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss


def region_cosine(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul', log=False):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)

    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_loco(model, dataloader, device, _class_=None, calc_pro=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            defect_type_list.extend(defect_type)

        gt_list_sp = np.array(gt_list_sp)
        pr_list_sp = np.array(pr_list_sp)
        defect_type_list = np.array(defect_type_list)
        auroc = roc_auc_score(gt_list_sp, pr_list_sp)
        auroc_logic = roc_auc_score(
            gt_list_sp[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            pr_list_sp[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            gt_list_sp[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            pr_list_sp[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_mean = (auroc_logic + auroc_struct) / 2

    return auroc, auroc_logic, auroc_struct, auroc_mean


def evaluation_mask(model, dataloader, device, _class_=None, calc_pro=True):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, mask, _ in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            if mask.max() > 0:
                # mask = binary_dilation(mask[0, 0].cpu().numpy().astype(int), iterations=2)
                mask = mask[0, 0].cpu().numpy().astype(int)
                anomaly_map = anomaly_map * mask
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


def evaluation_noseg(model, dataloader, device, _class_=None, reduction='max'):
    model.eval()
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt_list_sp.append(label.item())
            if reduction == 'max':
                pr_list_sp.append(np.max(anomaly_map))
            elif reduction == 'mean':
                pr_list_sp.append(np.mean(anomaly_map))

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh)
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_sp, f1, acc


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=-0.05, adaptive=False, **kwargs):
        assert rho <= 0.0, f"Invalid rho, should be negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
