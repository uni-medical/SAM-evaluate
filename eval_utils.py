from typing import Tuple
import torch
import numpy as np
import os
import skimage.io as io
from PIL import Image
import nibabel as nib
import logging
import torchvision
import cv2

def select_label(labels):

    labels = torch.tensor(labels, dtype=torch.int)

    if len(labels.shape) == 4:
        labels = labels.squeeze(1)

    if labels.shape[0] == 1:
        output_tensor = torch.as_tensor(labels[0:1], dtype=torch.int)
        keep_indices = torch.tensor([0], dtype=torch.long)
    else:
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
    
    mask = torch.tensor(np.array(mask), dtype=torch.int)
    label = torch.tensor(np.array(label), dtype=torch.int)

    if len(mask.shape) != 5 and mask.shape[2] != 1:
        mask = mask.unsqueeze(2)
    if len(label.shape) != 4 and label.shape[1] != 1:
        label = label.unsqueeze(1)

    # import pdb; pdb.set_trace()
    if mask.shape[-2:] != label.shape[-2:]:
        label = torch.nn.functional.interpolate(label.float(), size=mask.shape[-2:], mode='nearest')

    selected_mask = []
    overlap_score = []
    for i in range(label.shape[0]):  # Batch/class
        mask_ = mask[i]  # shape (n,1,h,w)
        label_ = label[i:i+1]  #shape (1,1,h,w)

        overlaps = overlap(mask_, label_)
        overlap_score.append(overlaps)
        idx = torch.argmax(overlaps, dim=0)
        selected_mask.append(mask_[idx])


    out_mask = torch.cat(selected_mask, dim=0) # shape (n, 1, h,w)
    out_overlap_score = torch.stack(overlap_score, dim=0)  # shape (b, n, 1)

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
        
def is_saved(save_path, mask_name, num_class):
    save_overlap_path = os.path.join(save_path, 'predict_masks')
    for i in range(num_class):
        if num_class == 1:
            save_name = mask_name
        else:
            save_name = mask_name.split('.nii.gz')[0] + '_' + str(i+1).zfill(3) + '.nii.gz'
        if not os.path.exists(os.path.join(save_overlap_path, save_name)):
            return False
    return True


def update_result_dict(save_path, mask_name, num_class, res_dict, metrics):
    if mask_name in res_dict:
        return res_dict
    
    save_overlap_path = os.path.join(save_path, 'predict_masks')
    gt_path = os.path.join(save_path, 'gt_segmentations')
    res_dict[mask_name] = {'iou': [], 'dice': []}
    for j in range(num_class):
        if num_class == 1:
            save_name = mask_name
        else:
            save_name = mask_name.split('.nii.gz')[0] + '_' + str(j+1).zfill(3) + '.nii.gz'

        itk_pred = sitk.ReadImage(os.path.join(save_overlap_path, save_name))
        itk_label = sitk.ReadImage(os.path.join(gt_path, save_name))
        pred_j = torch.tensor(sitk.GetArrayFromImage(itk_pred)).unsqueeze(1)
        label_j = torch.tensor(sitk.GetArrayFromImage(itk_label)).unsqueeze(1)
        if label_j.sum() > 0:
            slice_ids = torch.where(label_j)[0].unique()
            class_metric_j = SegMetrics(pred_j[slice_ids], label_j[slice_ids], metrics)
            res_dict[mask_name]['iou'].append(class_metric_j[0].item())
            res_dict[mask_name]['dice'].append(class_metric_j[1].item())
        else:
            res_dict[mask_name]['iou'].append(-1)
            res_dict[mask_name]['dice'].append(-1)
    return res_dict

def save_img(predict_score, predict_overlap, label, save_path, mask_name, class_idex, origin_size):
    predict_score = predict_score.cpu().numpy()
    class_idex = class_idex.cpu().numpy()

    if len(label.shape) > 3:
        label = label.squeeze(1).cpu().numpy()
    else:
        label = label.cpu().numpy()

    if len(predict_overlap.shape) > 3:
        predict_overlap = predict_overlap.squeeze(1).cpu().numpy()
    else:
        predict_overlap = predict_overlap.cpu().numpy()

    assert(label.shape[0]==predict_score.shape[0])
    assert(label.shape[0]==predict_overlap.shape[0])

    N, H, W = predict_score.shape

    save_score_path = os.path.join(save_path, 'score_masks')
    save_overlap_path = os.path.join(save_path, 'overlap_masks')
    save_label_path = os.path.join(save_path, 'masks')

    os.makedirs(save_score_path, exist_ok=True)
    os.makedirs(save_overlap_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    resize_score, resize_overlap, resize_label = [], [], []

    for i in range(N):
        try:
            resized_score_img = cv2.resize(predict_score[i], (origin_size[1], origin_size[0]), interpolation=cv2.INTER_LINEAR)
            resized_overlap_img = cv2.resize(predict_overlap[i], (origin_size[1], origin_size[0]), interpolation=cv2.INTER_LINEAR)
  
        except:
            resized_score_img = cv2.resize(predict_score[i], (origin_size[1], origin_size[0]), interpolation=cv2.INTER_NEAREST)
            resized_overlap_img = cv2.resize(predict_overlap[i], (origin_size[1], origin_size[0]), interpolation=cv2.INTER_NEAREST)
    
        save_name = ".".join(mask_name.split('.')[:-1]) + '_' + str(class_idex[i]+1).zfill(3) + '.png'

        cv2.imwrite(os.path.join(save_score_path, save_name), np.uint8(resized_score_img * 255))
        cv2.imwrite(os.path.join(save_overlap_path, save_name), np.uint8(resized_overlap_img* 255))
        cv2.imwrite(os.path.join(save_label_path, save_name), np.uint8(label[i]* 255))

        resize_score.append(torch.as_tensor(resized_score_img, dtype=torch.int))
        resize_overlap.append(torch.as_tensor(resized_overlap_img, dtype=torch.int))
        resize_label.append(torch.as_tensor(label[i], dtype=torch.int))


    # import pdb; pdb.set_trace()
    resize_scores = torch.stack(resize_score, dim=0).unsqueeze(1)
    resize_overlaps = torch.stack(resize_overlap, dim=0).unsqueeze(1)
    resize_labels = torch.stack(resize_label, dim=0).unsqueeze(1)
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
