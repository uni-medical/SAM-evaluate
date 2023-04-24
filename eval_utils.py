from typing import Tuple
import torch
import numpy as np
import os
import skimage.io as io
from PIL import Image
import nibabel as nib

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
        mask (tensor): tensor of shape [B, N, C, H, W] containing the masks to select from.
        iou (tensor): tensor of shape [B, N] containing the IoU scores.

    Returns:
        selected_mask (tensor): tensor of shape [B, C, H, W] containing the mask with the highest IoU score.
    """
    B, N, C, H, W = mask.shape
    max_iou, max_idx = torch.max(iou, dim=1, keepdim=True)
    max_idx = max_idx.long()
    selected_mask = torch.gather(mask, dim=1, index=max_idx.unsqueeze(2).unsqueeze(2).repeat(1, 1, 1, H, W))
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


def save_img3d(predict_score, predict_overlap, label, image, save_path, mask_name):

    if predict_score.dtype == torch.bool:
        predict_score = predict_score.type(torch.int)

    if predict_overlap.dtype == torch.bool:
        predict_overlap = predict_overlap.type(torch.int)

    predict_score = predict_score.cpu().numpy()     #C,1,256,256
    predict_overlap = predict_overlap.cpu().numpy() #C,1,256,256
    label = label.cpu().numpy()  #C,1,256,256
    image = image.cpu().numpy()[...,0]  #C,256,256,1

    predict_score = np.moveaxis(np.squeeze(predict_score, axis=1), [0, 1, 2], [2, 0, 1])
    predict_overlap = np.moveaxis(np.squeeze(predict_overlap, axis=1), [0, 1, 2], [2, 0, 1])
    label = np.moveaxis(np.squeeze(label, axis=1), [0, 1, 2], [2, 0, 1])
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])

    save_score_path = os.path.join(save_path, 'score_masks')
    save_overlap_path = os.path.join(save_path, 'overlap_masks')
    save_label_path = os.path.join(save_path, 'masks')
    save_image_path = os.path.join(save_path, 'images')

    os.makedirs(save_score_path, exist_ok=True)
    os.makedirs(save_overlap_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)
    os.makedirs(save_image_path, exist_ok=True)

    predict_score = nib.Nifti1Image(predict_score, affine=np.eye(4))
    predict_overlap = nib.Nifti1Image(predict_overlap, affine=np.eye(4))
    label = nib.Nifti1Image(label, affine=np.eye(4))
    image = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(predict_score, os.path.join(save_score_path, mask_name))
    nib.save(predict_overlap, os.path.join(save_overlap_path, mask_name))
    nib.save(label, os.path.join(save_label_path, mask_name))
    nib.save(image, os.path.join(save_image_path, mask_name))


def save_img(predict_score, predict_overlap, label, save_path, mask_name):

    if predict_score.dtype == torch.bool:
        predict_score = predict_score.type(torch.int)

    if predict_overlap.dtype == torch.bool:
        predict_overlap = predict_overlap.type(torch.int)

    predict_score = predict_score.cpu().numpy()
    predict_overlap = predict_overlap.cpu().numpy()
    label = label.cpu().numpy()

    N, C, H, W = predict_score.shape

    save_score_path = os.path.join(save_path, 'score_masks')
    save_overlap_path = os.path.join(save_path, 'overlap_masks')
    save_label_path = os.path.join(save_path, 'masks')

    os.makedirs(save_score_path, exist_ok=True)
    os.makedirs(save_overlap_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    score_paths = [os.path.join(save_score_path, name) for name in mask_name]
    overlap_paths = [os.path.join(save_overlap_path, name) for name in mask_name]
    label_paths = [os.path.join(save_label_path, name) for name in mask_name]

    for i in range(N):
        pr_score_img = np.squeeze(predict_score[i])
        pr_overlap_img = np.squeeze(predict_overlap[i])
        label_img = np.squeeze(label[i])

        pr_score_img = Image.fromarray(np.uint8(pr_score_img * 255))
        pr_overlap_img = Image.fromarray(np.uint8(pr_overlap_img * 255))
        label_img = Image.fromarray(np.uint8(label_img * 255))

        pr_score_img.save(score_paths[i])
        pr_overlap_img.save(overlap_paths[i])
        label_img.save(label_paths[i])
        #io.imsave(score_paths[i], (pr_score_img * 255).astype('uint8'))
        #io.imsave(overlap_paths[i], (pr_overlap_img * 255).astype('uint8'))
        #io.imsave(label_paths[i], (label_img * 255).astype('uint8'))


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