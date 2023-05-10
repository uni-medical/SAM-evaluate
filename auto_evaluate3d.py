from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from data_load3d import Data_Loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from eval_utils import save_img3d, SegMetrics, is_saved, update_result_dict
from evaluate3d import get_logger
import argparse
import torch.nn as nn
import logging
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.transform import resize
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image_size")
    parser.add_argument("--data_path", type=str, default='mount_preprocessed_sam/3d/semantic_seg/mr_de/Myops2020',help="eval data path")
    parser.add_argument("--dim", type=str, default='z', help="testing dim, default 'z' ")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[3,4,5,6], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_h", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAM-Med/pretrain_model/sam_vit_h.pth",help="sam checkpoint")
    parser.add_argument("--save_path", type=str, default='Evaluate-SAM/save_datasets/3d/mr_de/Myops2020', help="save data path")
    args = parser.parse_args()

    return args

def label_mask_overlap(label, mask, device):

    label = label.to(device)
    mask = mask.to(device)

    N = label.shape[0]  #N, H, W
    B = mask.shape[0]   #B, H, W
    # 计算每个通道的重叠率

    if mask.shape[-2:] != label.shape[-2:]:
        label = torch.nn.functional.interpolate(label.unsqueeze(1).float(), size=mask.shape[-2:], mode='nearest')
        label = label.squeeze(1)

    overlap = torch.zeros(B, N)        
    for i in range(B):
        for j in range(N):
            overlap[i][j] = torch.sum(torch.logical_and(label[j], mask[i])) / torch.sum(torch.logical_or(label[j], mask[i]))
    # 取出前N个最大的重叠率的索引
    indices = torch.argmax(overlap, dim=0).to(device)
    # 提取pred中的对应通道
    output = torch.index_select(mask, 0, indices)

    return output

def evaluate_batch_images(args, model):

    dataset = Data_Loader(data_path=args.data_path,
                          image_size=args.image_size,
                          prompt_point=False,
                          prompt_box=False,
                          num_boxes=0,
                          num_point=0,
                          dim = args.dim,
                          )

    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    progress_bar = tqdm(train_loader)
    dim = next(iter(progress_bar))['dim'][0]

    save_path = os.path.join(args.save_path, "auto_predict3d")
    print ('***** The save root is: {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)

    txt_path = save_path + "/{}.txt".format("auto_predict3d")

    json_path = os.path.join(save_path, 'result.json')

    if os.path.exists(json_path):
        res_dict = json.load(open(json_path, 'r'))
    else:
        res_dict = {}

    loggers = get_logger(txt_path)

    print(f"Volume number: {len(dataset)}")

    mean_iou, mean_dice = [], []
    for batch_input in progress_bar:
        image = batch_input['image'][0]   #(1, slice, class, 256, 256, 3) or [1, slice, 1, 256, 256, 3]
        label = batch_input['label'][0]
        ori_label = batch_input['ori_label'][0]
        zero_mask = batch_input['zero_mask'][0]
        index =  batch_input['index'][0]
        mask_name = batch_input['name'][0]

        if len(image.shape) == 1 and image.shape[0] == 1: # Filter cases with no foreground
            continue

        if is_saved(save_path, mask_name, ori_label.shape[1]):
            res_dict = update_result_dict(save_path, mask_name, ori_label.shape[1], res_dict, args.metrics)
            mean_iou.append(res_dict[mask_name]['iou'])
            mean_dice.append(res_dict[mask_name]['dice'])
            loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['iou'])} category IoU: {res_dict[mask_name]['iou']}")
            loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['dice'])} category Dice: {res_dict[mask_name]['dice']}")
            with open(json_path, 'w') as fid:
                json.dump(res_dict, fid, indent=4, sort_keys=True)
            continue

        volume_mask = []
        volume_dict = {}
        for i in range(image.shape[0]):  #slice
            for j in range(image.shape[1]): #class
                image_ = image[i][j].cpu().numpy().astype(np.uint8)
                masks = model.generate(image_)
                class_mask = [x['segmentation'] for x in masks]  #一个class

                assert(len(class_mask)>=1)

                masks = torch.as_tensor(np.stack(class_mask, axis=0), dtype=torch.int) # n, h, w
                
                ori_masks = torch.zeros((masks.shape[0], *ori_label.shape[-2:]), dtype=torch.int) # n, h, w
                for k in range(masks.shape[0]):
                    m = resize(masks[k].cpu().numpy().astype(float), ori_label.shape[-2:], 1, mode='edge', clip=True, anti_aliasing=False) # Follow nnUNet
                    ori_masks[k, m >= 0.5] = 1

                ori_label_j = ori_label[i, j].unsqueeze(0) # 1, h, w
                best_mask = label_mask_overlap(ori_label_j, ori_masks, args.device) # 1, h, w

                if j not in volume_dict:
                    volume_dict[j] = []
                volume_dict[j].append(best_mask) # key: n_class, val: 1, h, w

        res_dict[mask_name] = {'iou': [], 'dice': []}
        for j in range(ori_label.shape[1]):
            volume_mask.append(torch.stack(volume_dict[j], dim=1)) # 1, n_slice, h, w
            label_j = ori_label[:, j:j+1, ...] # n_slice, 1, h, w
            if label_j.sum() > 0:
                slice_ids = torch.where(label_j)[0].unique()
                class_metric_j = SegMetrics(torch.stack(volume_dict[j], dim=0).cpu()[slice_ids], label_j.cpu()[slice_ids], args.metrics)
                res_dict[mask_name]['iou'].append(class_metric_j[0].item())
                res_dict[mask_name]['dice'].append(class_metric_j[1].item())
            else:
                res_dict[mask_name]['iou'].append(-1)
                res_dict[mask_name]['dice'].append(-1)
        
        mean_iou.append(res_dict[mask_name]['iou'])
        mean_dice.append(res_dict[mask_name]['dice'])
        
        loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['iou'])} category IoU: {res_dict[mask_name]['iou']}")
        loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['dice'])} category Dice: {res_dict[mask_name]['dice']}")
        
        # import pdb; pdb.set_trace()
        save_img3d(torch.cat(volume_mask, dim=0), save_path, batch_input['name'][0], zero_mask, index, ori_label)
        with open(json_path, 'w') as fid:
            json.dump(res_dict, fid, indent=4, sort_keys=True)

    mean_iou = np.array(mean_iou) # n_case, n_class
    mean_dice = np.array(mean_dice) # n_case, n_class
    mean_iou[np.where(mean_iou == -1)] = np.nan
    mean_dice[np.where(mean_dice == -1)] = np.nan
    iou = np.nanmean(mean_iou, axis=0)
    dice = np.nanmean(mean_dice, axis=0)
    for i in range(len(iou)):
        iou[i] = '{:.4f}'.format(iou[i])
        dice[i] = '{:.4f}'.format(dice[i])
    loggers.info(f"{len(mean_iou)} volume mIoU: {iou}")
    loggers.info(f"{len(mean_dice)} volume mDice: {dice}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    args.device = device
    model = sam_model_registry[args.model_type](args.sam_checkpoint, args.image_size)
    if len(args.device_ids) > 1:
        print("Let's use", len(args.device_ids), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.device_ids)
        predictor = SamAutomaticMaskGenerator(  
            model.module.to(args.device),
            points_per_side = 32,
            points_per_batch = 64,
            pred_iou_thresh = 0.80,
            stability_score_thresh = 0.90,
            stability_score_offset = 1.0,
            box_nms_thresh = 0.5,
            crop_n_layers = 0,
            crop_nms_thresh = 0.7,
            crop_overlap_ratio = 512 / 1500,
            crop_n_points_downscale_factor = 1,
            min_mask_region_area = 4,
            output_mode = "binary_mask",
                                            )
    else:
        predictor = SamAutomaticMaskGenerator(            
            model.to(args.device),
            points_per_side = 32,
            points_per_batch = 64,
            pred_iou_thresh = 0.80,
            stability_score_thresh = 0.90,
            stability_score_offset = 1.0,
            box_nms_thresh = 0.5,
            crop_n_layers = 0,
            crop_nms_thresh = 0.7,
            crop_overlap_ratio = 512 / 1500,
            crop_n_points_downscale_factor = 1,
            min_mask_region_area = 4,
            output_mode = "binary_mask",
            )
        
    evaluate_batch_images(args, predictor)