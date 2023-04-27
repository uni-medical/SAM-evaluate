from typing import Tuple
import torch
import numpy as np
import os
import skimage.io as io
from PIL import Image
import nibabel as nib
import logging
import torchvision


def select_label(labels):
    if len(labels.shape) == 4:
        labels = labels.squeeze(1)

    channel_sums = labels.sum(dim=(1, 2))
    all_zeros = (channel_sums == 0.0)
    keep_indices = torch.arange(labels.shape[0])[~all_zeros]
    output_tensor = labels[keep_indices, ...]
    return output_tensor, keep_indices


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



def move_batch_to_device(batch_input, device):
    """将批量输入数据移动到指定设备上（如GPU）"""
    if batch_input.get("image") is not None:
        batch_input["image"] = batch_input["image"].float().to(device)
    if batch_input.get("label") is not None:
        batch_input["label"] = batch_input["label"].float().to(device)
    if batch_input.get("point_coords") is not None:
        batch_input["point_coords"] = batch_input["point_coords"].to(device)
    if batch_input.get("point_labels") is not None:
        batch_input["point_labels"] = batch_input["point_labels"].to(device)
    if batch_input.get("boxes") is not None:
        batch_input["boxes"] = batch_input["boxes"].to(device)
    if batch_input.get("mask_inputs") is not None:
        batch_input["mask_inputs"] = batch_input["mask_inputs"].to(device)

    return batch_input

def select_mask_with_highest_iouscore(mask, iou):
    """
    Returns the mask with highest IoU score.

    Args:
        mask (tensor): tensor of shape [B, N, H, W] containing the masks to select from.
        iou (tensor): tensor of shape [B, N] containing the IoU scores.

    Returns:
        selected_mask (tensor): tensor of shape [B, C, H, W] containing the mask with the highest IoU score.
    """
    B, N, H, W = mask.shape
    max_iou, max_idx = torch.max(iou, dim=1, keepdim=True)
    max_idx = max_idx.long()
    selected_mask = torch.gather(mask, dim=1, index=max_idx.unsqueeze(2).unsqueeze(2).repeat(1, 1, H, W))
    selected_mask = selected_mask.squeeze(1)
    return selected_mask


def overlap(preds, masks):
    overlaps = []
    for i in range(preds.shape[0]):
        overlap = []  # 每个预测结果与所有真值图的重叠情况
        for j in range(masks.shape[0]):
            intersection = (preds[i] * masks[j]).sum()  # 计算交集
            union = (preds[i] + masks[j]).sum() - intersection  # 计算并集
            overlap.append(intersection / union)  # 计算重叠情况
        overlaps.append(overlap)

    overlaps = torch.stack([torch.tensor(overlaps[i]) for i in range(len(overlaps))])  # 转换为张量形式
    return overlaps


def select_mask_with_highest_overlap(mask: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        mask: shape (b,n,1,h,w)
        label: shape (b,1,h,w)

    Returns:
        out_mask: shape (n,h,w)
        out_overlap_score: shape (b,n)
    """


    selected_mask = []
    overlap_score = []
    for i in range(label.shape[0]):  # Batch
        mask_ = mask[i]  # shape (n,1,h,w)
        label_ = label[i:i+1]
        overlaps = overlap(mask_, label_)
        overlap_score.append(overlaps)
        idx = torch.argmax(overlaps, dim=0)
        selected_mask.append(mask_[idx])

    out_mask = torch.cat(selected_mask, dim=0) # shape (n,h,w)
    out_overlap_score = torch.stack(overlap_score, dim=0)  # shape (b,n)

    return out_mask, out_overlap_score


def save_img3d(predict_overlap, save_path, mask_name, zero_mask, index):   #zero_mask [168, 4, 256, 256]   index [67]

    if predict_overlap.dtype == torch.bool:
        predict_overlap = predict_overlap.type(torch.int) #class, slice, 256, 256

    predict_overlap = predict_overlap.cpu().numpy()
    zero_mask = zero_mask.cpu().numpy()

    save_overlap_path = os.path.join(save_path, 'predict_masks')
    os.makedirs(save_overlap_path, exist_ok=True)

    for i in range(predict_overlap.shape[0]): #class
        pred = predict_overlap[i]
        pred_ = np.moveaxis(pred, [0, 1, 2], [2, 0, 1])  #256, 256, 67
        class_zero_mask = np.moveaxis(zero_mask[:,i,...], [0, 1, 2], [2, 0, 1])  #256, 256, 168
        for item, value in enumerate(index):
            class_zero_mask[:, :, value] = pred_[:, :, item]

        if predict_overlap.shape[0] == 1:
            save_name = mask_name
        else:
            save_name = mask_name.split('.nii.gz')[0] + '_' + str(i+1).zfill(3) + '.nii.gz'

        predict = nib.Nifti1Image(class_zero_mask, affine=np.eye(4))
        nib.save(predict, os.path.join(save_overlap_path, save_name))


def save_img(predict_score, predict_overlap, label, save_path, mask_name, class_idex, origin_size):
    predict_score = predict_score.cpu().numpy()
    predict_overlap = predict_overlap.cpu().numpy()
    label = label.cpu().numpy()
    class_idex = class_idex.cpu().numpy()

    N, H, W = predict_score.shape

    save_score_path = os.path.join(save_path, 'score_masks')
    save_overlap_path = os.path.join(save_path, 'overlap_masks')
    save_label_path = os.path.join(save_path, 'masks')

    os.makedirs(save_score_path, exist_ok=True)
    os.makedirs(save_overlap_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    trans_totensor = torchvision.transforms.ToTensor()
    resize_score, resize_overlap, resize_label = [], [], []
    for i in range(N):
        score_img = Image.fromarray(np.uint8(predict_score[i] * 255))
        overlap_img = Image.fromarray(np.uint8(predict_overlap[i] * 255))
        label_img = Image.fromarray(np.uint8(label[i] * 255))
        
        resized_score_img = score_img.resize(origin_size, resample=Image.NEAREST)
        resized_overlap_img = overlap_img.resize(origin_size, resample=Image.NEAREST)
        resized_label_img = label_img.resize(origin_size, resample=Image.NEAREST)

        save_name = mask_name.split('.')[0] + '_' + str(class_idex[i]+1).zfill(3) + '.png'
        resized_score_img.save(os.path.join(save_score_path, save_name))
        resized_overlap_img.save(os.path.join(save_overlap_path, save_name))
        resized_label_img.save(os.path.join(save_label_path, save_name))

        resize_score.append(trans_totensor(resized_score_img))
        resize_overlap.append(trans_totensor(resized_overlap_img))
        resize_label.append(trans_totensor(resized_label_img))

    resize_scores = torch.stack(resize_score, dim=0).unsqueeze(1) / 255.
    resize_overlaps = torch.stack(resize_overlap, dim=0).unsqueeze(1) / 255.
    resize_labels = torch.stack(resize_label, dim=0).unsqueeze(1) / 255.
    return resize_scores, resize_overlaps, resize_labels

def iou(pr, gt, eps=1e-7):

    intersection = torch.sum(pr * gt, dim=[1,2,3])
    union = torch.sum(gt,dim=[1,2,3]) + torch.sum(pr,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()

def dice(pr, gt, eps=1e-7):
    intersection = torch.sum(gt * pr, dim=[1,2,3])
    union = torch.sum(gt,dim=[1,2,3]) + torch.sum(pr,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def f_score(pr, gt, beta=1, eps=1e-7):
    tp = torch.sum(gt * pr,dim=[1,2,3])
    fp = torch.sum(pr,dim=[1,2,3]) - tp
    fn = torch.sum(gt,dim=[1,2,3]) - tp
    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score.cpu().numpy()

def acc(pr, gt):
    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score.cpu().numpy()

def precision(pr, gt, eps=1e-7):
    tp = torch.sum(gt * pr,dim=[1,2,3])
    fp = torch.sum(pr,dim=[1,2,3]) - tp
    score = (tp + eps) / (tp + fp + eps)
    return score.cpu().numpy()

def recall(pr, gt, eps=1e-7):
    tp = torch.sum(gt * pr,dim=[1,2,3])
    fn = torch.sum(gt,dim=[1,2,3]) - tp
    score = (tp + eps) / (tp + fn + eps)
    return score.cpu().numpy()
sensitivity = recall


def spec(pr, gt, eps=1e-7):
    true_neg = torch.sum((1-gt) * (1-pr), dim=[1,2,3])
    total_neg = torch.sum((1-gt), dim=[1, 2, 3])
    score = (true_neg + eps) / (total_neg + eps)
    return score.cpu().numpy()




def SegMetrics(pred, label, metrics):
    metric_list = []

    if pred.dtype == torch.bool:
        pred = pred.type(torch.int)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metric_list.append(np.mean(acc(pred, label)))
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'prec':
            metric_list.append(np.mean(precision(pred, label)))
        elif metric == 'sens':
            metric_list.append(np.mean(recall(pred, label)))
        elif metric == 'spec':
            metric_list.append(np.mean(spec(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)

    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')

    return metric
