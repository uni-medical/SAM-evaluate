from segment_anything import sam_model_registry, SamPredictor
from data_load3d import Data_Loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from eval_utils import save_img3d, select_mask_with_highest_iouscore, select_mask_with_highest_overlap, SegMetrics
import argparse
import torch.nn as nn
import logging
import os
import numpy as np
from matplotlib import pyplot as plt
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--data_path", type=str, default='datasets/ISLES2018/',help="eval data path")
    parser.add_argument("--dim", type=str, default='z', help="testing dim, default 'z' ")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0,1], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_h", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam_vit_h_4b8939.pth",help="sam checkpoint")
    parser.add_argument("--include_prompt_point", type=bool, default=True, help="need point prompt")
    parser.add_argument("--num_point", type=int, default=1, help="point or point number")
    parser.add_argument("--include_prompt_box", type=bool, default=True, help="need boxes prompt")
    parser.add_argument("--num_boxes", type=int, default=1, help="boxes or boxes number")
    parser.add_argument("--multimask_output", type=bool, default=True, help="multimask output")
    parser.add_argument("--save_path", type=str, default='save_datasets/3d/ISLES2018/',
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

    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=32)
    progress_bar = tqdm(train_loader)
    dim = next(iter(progress_bar))['dim'][0]

    save_path = os.path.join(args.save_path,
                             f"{dim}_{args.image_size}"
                             f"_{'boxes' if args.include_prompt_box else 'points'}"
                             f"_{args.num_boxes if args.include_prompt_box else args.num_point}")

    os.makedirs(save_path, exist_ok=True)

    txt_path = os.path.join(save_path,
                            f"{dim}_{args.image_size}"
                            f"_{'boxes' if args.include_prompt_box else 'points'}"
                            f"_{args.num_boxes if args.include_prompt_box else args.num_point}.txt")

    loggers = get_logger(txt_path)

    print(f"Volume number: {len(dataset)}")

    mean_iou, mean_dice = [], []
    for batch_input in progress_bar:

        image = batch_input['image'][0]   #(1, slice, class, 256, 256, 3) or [1, slice, 1, 256, 256, 3]
        label = batch_input['label'][0]

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

        slice_iou = [0] * image.shape[1]
        slice_dice = [0] * image.shape[1]
        volume_mask, volume_label = [],[]
        for i in range(image.shape[0]):  #slice
            class_mask= []
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

                masks = torch.as_tensor(masks, dtype=torch.int)

                if len(masks) == 3:
                    masks = masks.unsqueeze(1)

                elif len(masks) == 4:
                    masks = masks

                class_mask.append(masks)

            class_out_mask = torch.stack(class_mask, dim=0).to(device)  #class, 1, h, w

            label_ = label[i].unsqueeze(1).to(device)      #class, 1, h, w

            best_masks, overlap_score = select_mask_with_highest_overlap(class_out_mask, label_)
            volume_mask.append(best_masks)

            for x in range(best_masks.shape[0]):
                class_metrics_ = SegMetrics(best_masks[x : x+1], label_[x : x+1], args.metrics)  #1 slice产生class个特征图
                slice_iou[x] += class_metrics_[0]
                slice_dice[x] += class_metrics_[1]

        if len(slice_iou) == 1:
            save_img3d(torch.cat(volume_mask, dim=0), label, image[:,0,...], save_path, batch_input['name'][0])

        for y in range(len(slice_iou)):
            slice_iou[y] /= image.shape[0]
            slice_dice[y] /= image.shape[0]

        loggers.info(f"{batch_input['name'][0]} volume {len(slice_iou)} category IoU: {slice_iou}")
        loggers.info(f"{batch_input['name'][0]} volume {len(slice_dice)} category Dice: {slice_dice}")

        mean_iou.append(slice_iou)
        mean_dice.append(slice_dice)


    iou = np.array(mean_iou).mean(axis=0)
    dice = np.array(mean_dice).mean(axis=0)
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
    evaluate_batch_images(args, predictor)
