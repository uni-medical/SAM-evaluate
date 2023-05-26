from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from data_load3d import Data_Loader
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import argparse
import torch.nn as nn
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.transform import resize
import json
import glob
import nibabel as nib
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--data_path", type=str,default='/home/chengjunlong/mount_preprocessed_sam/2d/semantic_seg/fundus_photography/gamma3/',help="eval data path")
    parser.add_argument("--data_mode", type=str, default='train', help="eval train or test data")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[2, 3, 4], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAM-Med/pretrain_model/sam_vit_b.pth",help="sam checkpoint")
    parser.add_argument("--save_path", type=str, default='Evaluate-SAM/save_datasets/semantic_seg_2d_everything',help="save data path")
    args = parser.parse_args()
    return args


class everything_2d():
    def __init__(self, args, model):

        self.model = model
        self.save_path = args.save_path
        self.data_path = args.data_path
        self.mode = args.data_mode

        print('***** The save root is: {}'.format(self.save_path))
        os.makedirs(self.save_path, exist_ok=True)

        self.list_path = self.save_path + '/everything_list.txt'
        self.skip_path = self.save_path + "/skip_list.txt"

    def load_data(self, data_path):
        img_list = [line.strip() + ".png" for line in open(os.path.join(data_path, self.mode + ".txt"), "r").readlines()]
        imgs_path = [os.path.join(data_path, "images", img) for img in img_list]
        label_path = [os.path.join(data_path, "masks", img) for img in img_list]

        print(f"Volume number: {len(imgs_path)}")
        assert (len(imgs_path) == len(label_path)), 'Image and label data are not equal'
        imgs_path.sort()
        label_path.sort()

        return imgs_path, label_path

    def process_img(self, image, mask):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, 0)

        if np.all(mask == 0):
            print(f'{mask} unique == {np.unique(mask)}')
            ori_mask = None
        else:
            if set(np.unique(mask)) == {0, 255}:
                mask = mask / 255
            one_hot = np.eye(len(np.unique(mask)))[mask]
            ori_mask = one_hot.transpose(2, 0, 1)[1:]

        return image, ori_mask

    def generate_mask(self, model, input_image, image_name=None):
        generate_masks = model.generate(input_image)

        everything_mask = [x['segmentation'] for x in generate_masks]  # 一个slice
        predicted_iou = [x['predicted_iou'] for x in generate_masks]  # 一个slice

        if len(everything_mask) >= 1:
            gen_masks = torch.as_tensor(np.stack(everything_mask, axis=0), dtype=torch.int)  # n, ori_h, ori_w
            scores = torch.as_tensor(np.stack(predicted_iou, axis=0), dtype=torch.float)  # n

        else:
            print(f"{image_name}  did not generate any mask")
            gen_masks = None
            scores = None

        return gen_masks, scores

    def save_img(self, masks, scores, ori_label, path):
        predict_mask = masks.cpu().numpy()
        score = scores.cpu().numpy()
        N, H, W = predict_mask.shape

        path_list = path.split('/')
        mask_name = path_list[-1].replace(".png","")
        folder = '/'.join(path_list[:-1])

        os.makedirs(folder, exist_ok=True)

        for i in range(N):
            save_name = mask_name + '_' + str(i + 1).zfill(3) + '_' + str('{:.4f}'.format(score[i])) + '.png'
            cv2.imwrite(os.path.join(folder, save_name), np.uint8(predict_mask[i] * 255))

            if os.path.exists(self.list_path):
                with open(self.list_path, 'a') as f:
                    f.write(os.path.join(folder, save_name) + '\n')
            else:
                with open(self.list_path, 'w') as f:
                    f.write(os.path.join(folder, save_name) + '\n')


        if ori_label is not None:
            X, H, W = ori_label.shape
            Y, h, w = predict_mask.shape

            assert H == h
            assert W == w

            skip_list = []
            for x in range(X):
                for y in range(Y):
                    intersection = (ori_label[x] * predict_mask[y]).sum()
                    union = ori_label[x].sum() + predict_mask[y].sum() - intersection
                    overlap = intersection / union
                    if overlap > 0.5:
                        skip_list.append(y)

            unique_list = list(set(skip_list))
            if len(unique_list) > 0:
                for i in unique_list:
                    skip_name = mask_name + '_' + str(i + 1).zfill(3) + '_' + str('{:.4f}'.format(score[i])) + '.png'

                    if os.path.exists(self.skip_path):
                        with open(self.skip_path, 'a') as f:
                            f.write(os.path.join(folder, skip_name) + '\n')
                    else:
                        with open(self.skip_path, 'w') as f:
                            f.write(os.path.join(folder, skip_name) + '\n')

    def is_save(self, exist_path):
        if not os.path.exists(exist_path):
            return False
        else:
            return True

    def predict(self):
        #mount_preprocessed_sam/2d/semantic_seg/fundus_photography/gamma3/

        imgs_path, label_path = self.load_data(self.data_path)
        exist_list = self.data_path.split('/2d/semantic_seg/')
        exist_list[0] = self.save_path
        exist_path = os.path.join('/'.join(exist_list), f"masks/{self.mode}") 

        if not self.is_save(exist_path):  # 判断数据集是否被保存过  Evaluate-SAM/save_datasets/semantic_seg_2d_everything/fundus_photography/gamma3/masks/train
            for item, imgs_path in enumerate(tqdm(imgs_path)):
                label_name = label_path[item].split('/')[-1]
                img_save_path = os.path.join(exist_path, label_name)

                image, ori_label = self.process_img(imgs_path, label_path[item])
                #(ori_h, ori_w, 3)(class , ori_h, ori_w)

                # Evaluate-SAM/save_datasets/semantic_seg_2d_everything/fundus_photography/gamma3/mode/labels/gamma_01.png
                everything_ori_masks, predicted_iou = self.generate_mask(self.model, image, imgs_path)
                if everything_ori_masks != None:
                    self.save_img(everything_ori_masks, predicted_iou, ori_label, img_save_path)



if __name__ == "__main__":
    args = parse_args()
    device_str = ','.join(str(i) for i in args.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device_str}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = sam_model_registry[args.model_type](args.sam_checkpoint, args.image_size)

    print("Let's use", len(args.device_ids), "GPUs!")

    predictor = SamAutomaticMaskGenerator(
        model.to(args.device),  # .module
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )

    eval_3d = everything_2d(args, predictor)
    eval_3d.predict()
