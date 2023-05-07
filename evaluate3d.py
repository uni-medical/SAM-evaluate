from segment_anything import sam_model_registry, SamPredictor
from data_load3d import Data_Loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from eval_utils import save_img3d, select_mask_with_highest_iouscore, select_mask_with_highest_overlap, SegMetrics, is_saved, update_result_dict
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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--data_path", type=str, default='datasets/autoPET/',help="eval data path")
    parser.add_argument("--dim", type=str, default='z', help="testing dim, default 'z' ")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0,1], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_h", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam_vit_h_4b8939.pth",help="sam checkpoint")
    parser.add_argument("--include_prompt_point", type=str2bool, default=True, help="need point prompt")
    parser.add_argument("--num_point", type=int, default=1, help="point or point number")
    parser.add_argument("--include_prompt_box", type=str2bool, default=True, help="need boxes prompt")
    parser.add_argument("--num_boxes", type=int, default=1, help="boxes or boxes number")
    parser.add_argument("--multimask_output", type=str2bool, default=True, help="multimask output")
    parser.add_argument("--save_path", type=str, default='save_datasets/3d/autoPET/',
                        help="save data path")
    args = parser.parse_args()

    return args


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


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


def evaluate_batch_images(args, model):

    dataset = Data_Loader(data_path=args.data_path,
                          image_size=args.image_size,
                          prompt_point=args.include_prompt_point,
                          prompt_box=args.include_prompt_box,
                          num_boxes=args.num_boxes,
                          num_point=args.num_point,
                          dim = args.dim,
                          )

    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    progress_bar = tqdm(train_loader)
    dim = next(iter(progress_bar))['dim'][0]

    save_path = os.path.join(args.save_path,
                             f"{dim}_{args.image_size}"
                             f"_{'boxes' if args.include_prompt_box else 'points'}"
                             f"_{args.num_boxes if args.include_prompt_box else args.num_point}")
    print ('***** The save root is: {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)

    txt_path = os.path.join(save_path,
                            f"{dim}_{args.image_size}"
                            f"_{'boxes' if args.include_prompt_box else 'points'}"
                            f"_{args.num_boxes if args.include_prompt_box else args.num_point}.txt")
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
            continue

        if args.include_prompt_point:
            point_coord = batch_input['point_coords'][0]  #[1,slice, class, N, 2])
            point_label = batch_input['point_labels'][0]
        else:
            point_coord = None
            point_label = None

        if args.include_prompt_box:
            boxs = batch_input['boxes'][0]
        else:
            boxs = None

        volume_mask = []
        volume_dict = {}
        for i in range(image.shape[0]):  #slice
            for j in range(image.shape[1]): #class
                image_ = image[i][j].cpu().numpy().astype(np.uint8)
                model.set_image(image_)

                if args.include_prompt_point:
                    point_coords = point_coord[i][j].cpu().numpy()
                    point_labels = point_label[i][j].cpu().numpy()

                else:
                    point_coords, point_labels = None, None

                if args.include_prompt_box:
                    if args.num_boxes > 1:
                        box = torch.as_tensor(boxs[i][j]).to(args.device)
                        box = model.transform.apply_boxes_torch(box, image.shape[:2])
                    else:
                        box = boxs[i][j].cpu().numpy()[0]
                else:
                    box = None

                if args.include_prompt_box and args.num_boxes > 1:
                    masks, scores, logits = model.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=box,
                        multimask_output=False,
                    )

                else:
                    masks, scores, logits = model.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=args.multimask_output,
                    )

                masks = torch.as_tensor(masks, dtype=torch.int) # 3, h, w
                
                ori_masks = torch.zeros((masks.shape[0], *ori_label.shape[-2:]), dtype=torch.int) # 3, h, w
                for k in range(masks.shape[0]):
                    m = resize(masks[k].cpu().numpy().astype(float), ori_label.shape[-2:], 1, mode='edge', clip=True, anti_aliasing=False) # Follow nnUNet
                    ori_masks[k, m >= 0.5] = 1

                if len(ori_masks) == 3:
                    ori_masks = ori_masks.unsqueeze(1) # 3, 1, h, w
                elif len(ori_masks) == 4:
                    ori_masks = ori_masks

                # class_mask.append(ori_masks)
                ori_label_j = ori_label[i, j].unsqueeze(0).unsqueeze(0).to(device) # 1, 1, h, w
                best_mask, overlap_score = select_mask_with_highest_overlap(ori_masks.to(device), ori_label_j) # 1, 1, h, w

                if j not in volume_dict:
                    volume_dict[j] = []
                volume_dict[j].append(best_mask) # key: n_class, val: 1, 1, h, w

        res_dict[mask_name] = {'iou': [], 'dice': []}
        for j in range(ori_label.shape[1]):
            volume_mask.append(torch.stack(volume_dict[j], dim=1)) # 1, n_slice, h, w
            label_j = ori_label[:, j:j+1, ...] # n_slice, 1, h, w
            if label_j.sum() > 0:
                slice_ids = torch.where(label_j)[0].unique()
                class_metric_j = SegMetrics(torch.stack(volume_dict[j], dim=0).cpu()[slice_ids], label_j.cpu()[slice_ids], args.metrics)
                res_dict[mask_name]['iou'].append(class_metric_j[0])
                res_dict[mask_name]['dice'].append(class_metric_j[1])
            else:
                res_dict[mask_name]['iou'].append(-1)
                res_dict[mask_name]['dice'].append(-1)
        
        mean_iou.append(res_dict[mask_name]['iou'])
        mean_dice.append(res_dict[mask_name]['dice'])
        
        loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['iou'])} category IoU: {res_dict[mask_name]['iou']}")
        loggers.info(f"{batch_input['name'][0]} volume {len(res_dict[mask_name]['dice'])} category Dice: {res_dict[mask_name]['dice']}")
        
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
        predictor = SamPredictor(model.module.to(args.device))
    else:
        predictor = SamPredictor(model.to(args.device))
    evaluate_batch_images(args, predictor)
