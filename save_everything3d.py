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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--data_path", type=str, default='mount_preprocessed_sam/3d/semantic_seg/mr_de/Myops2020',help="eval data path")
    parser.add_argument("--dim", type=str, default='z', help="testing dim, default 'z' ")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[2,3,4], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAM-Med/pretrain_model/sam_vit_b.pth",help="sam checkpoint")
    parser.add_argument("--save_path", type=str, default='Evaluate-SAM/save_datasets/semantic_seg_3d_everything', help="save data path")
    args = parser.parse_args()

    return args


class everything_3d():
    def __init__(self, args, model):

        self.model = model
        self.save_path = args.save_path
        self.dim = args.dim
        self.image_size = args.image_size
        self.data_path = args.data_path

        with open(os.path.join(self.data_path, 'dataset.json'), 'r') as f:
            data = json.load(f)
        self.num_class = len(data['labels'])

        print ('***** The save root is: {}'.format(self.save_path))
        os.makedirs(self.save_path, exist_ok=True)

        self.list_path = self.save_path + '/everything_list.txt'
        self.skip_path = self.save_path + "/skip_list.txt"
        self.json_path = os.path.join(self.save_path, 'result.json')

        if os.path.exists(self.json_path):
            self.res_dict = json.load(open(self.json_path, 'r'))
        else:
             self.res_dict = {}

    def load_data(self, data_path):
        imgs_path = []
        label_path = []
        if os.path.exists(os.path.join(data_path, 'labelsTs')):
            imgs_ts = glob.glob(os.path.join(data_path, 'imagesTs/*.nii.gz'))
            label_ts = glob.glob(os.path.join(data_path, 'labelsTs/*.nii.gz'))
            imgs_path  =  imgs_path + imgs_ts
            label_path = label_path + label_ts
        elif os.path.exists(os.path.join(data_path, 'labelsTr')):
            imgs_tr = glob.glob(os.path.join(data_path, 'imagesTr/*.nii.gz'))
            label_tr = glob.glob(os.path.join(data_path, 'labelsTr/*.nii.gz'))
            imgs_path  =  imgs_path + imgs_tr
            label_path = label_path + label_tr
        elif os.path.exists(os.path.join(data_path, 'labelsVal')):
            imgs_val = glob.glob(os.path.join(data_path, 'imagesVal/*.nii.gz'))
            label_val = glob.glob(os.path.join(data_path, 'labelsVal/*.nii.gz'))
            imgs_path  =  imgs_path + imgs_val
            label_path = label_path + label_val
        else:
            print('No data available for this dataset')

        print(f"Volume number: {len(imgs_path)}")
        assert (len(imgs_path)== len(label_path)), 'Image and label data are not equal'
        imgs_path.sort()
        label_path.sort()

        with open(os.path.join(data_path, 'dataset.json'), 'r') as f:
            data = json.load(f)
        class_num = len(data['labels'])

        return imgs_path, label_path, class_num
    
    def process_img(self, image, mask, image_size, dim, class_num):
        if dim == 'z':
            nonzero_slices = np.where(np.any(mask, axis=(0, 1)))[0]
            select_image = image[:, :, nonzero_slices]
            select_mask = mask[:, :, nonzero_slices]

        else:
            if dim == 'x':
                nonzero_slices = np.where(np.any(mask, axis=(1, 2)))[0]
                select_image = image[nonzero_slices, :, :].transpose(1,2,0)
                select_mask = mask[nonzero_slices, :, :].transpose(1,2,0)

            else:
                nonzero_slices = np.where(np.any(mask, axis=(0, 2)))[0]
                select_image = image[:, nonzero_slices, :].transpose(0,2,1)
                select_mask = mask[:,nonzero_slices, :].transpose(0,2,1)

        target_size = (image_size, image_size)
        max_pixel = select_image.max()
        min_pixel = select_image.min()
        select_image = (255 * (select_image - min_pixel) / (max_pixel - min_pixel)).astype(np.uint8)
  
        resized_image = cv2.resize(select_image, target_size, cv2.INTER_NEAREST)  #resize (H,W,1) -> (H,W)
        ori_mask = select_mask.astype(np.int16)

        if len(resized_image.shape) == 2:
                resized_image = resized_image[:,:,np.newaxis]

        volume_image = []
        volume_ori_mask = []
        for i in range(resized_image.shape[-1]):
            volume_image.append(np.repeat(resized_image[...,i:i+1], repeats=3, axis=-1))
            volume_ori_mask.append(ori_mask[...,i])

        volume_images = np.stack(volume_image, axis=0)
        volume_ori_masks = np.stack(volume_ori_mask, axis=0)

        if class_num == 2:
            ori_label = np.expand_dims(volume_ori_masks, axis=1)
        else:
            eye = np.eye(class_num, dtype=volume_ori_masks.dtype)
            ori_label = eye[volume_ori_masks].transpose(0, 3, 1, 2)[:,1:,...]
    
        return volume_images, ori_label

    def generate_mask(self, model, input_image, ori_label, image_name=None, slice = None):
        generate_masks = model.generate(input_image)
   
        everything_mask = [x['segmentation'] for x in generate_masks]  #一个slice
        predicted_iou = [x['predicted_iou'] for x in generate_masks]  #一个slice
      
        if  len(everything_mask)>= 1:
            masks = torch.as_tensor(np.stack(everything_mask, axis=0), dtype=torch.int) # n, h, w
            scores = torch.as_tensor(np.stack(predicted_iou, axis=0), dtype=torch.float) # n
        
            ori_masks = torch.zeros((masks.shape[0], *ori_label.shape[-2:]), dtype=torch.int) # n, ori_h, ori_w
            for k in range(masks.shape[0]):
                m = resize(masks[k].cpu().numpy().astype(float), ori_label.shape[-2:], 1, mode='edge', clip=True, anti_aliasing=False) # Follow nnUNet
                ori_masks[k, m >= 0.5] = 1
        else:
            print(f"{image_name} slice {slice} did not generate any mask")
            ori_masks =None
            scores = None

        return ori_masks, scores

    
    def save_img(self, masks, scores, ori_label, path):
        predict_mask = masks.cpu().numpy()
        score = scores.cpu().numpy()
        N, H, W = predict_mask.shape

        path_list = path.split('/')
        folder = '/'.join(path_list[:-1])
        mask_name = path_list[-1]
        

        os.makedirs(folder, exist_ok=True)
        for i in range(N):
            save_name = mask_name + '_' + str(i+1).zfill(3) + '_' + str('{:.4f}'.format(score[i])) + '.png'
            cv2.imwrite(os.path.join(folder, save_name), np.uint8(predict_mask[i] * 255))

            if os.path.exists(self.list_path):
                with open(self.list_path, 'a') as f:
                    f.write(os.path.join(folder, save_name) + '\n')
            else:
                with open(self.list_path, 'w') as f:
                    f.write(os.path.join(folder, save_name) + '\n')


        X, H, W = ori_label.shape
        Y, h, w = predict_mask.shape

        assert H==h
        assert W ==w

        skip_list = []
        for x in range(X):
            for y in range(Y):
                intersection = (ori_label[x] * predict_mask[y]).sum()
                union = ori_label[x].sum() + predict_mask[y].sum() - intersection
                overlap = intersection / union
                if overlap > 0.5:
                    skip_list.append(y)

        unique_list = list(set(skip_list))
        if len(unique_list)>0:
            for i in unique_list:
                skip_name = mask_name + '_' + str(i+1).zfill(3) + '_' + str('{:.4f}'.format(score[i])) + '.png'
            if os.path.exists(self.skip_path):
                with open(self.skip_path, 'a') as f:
                    f.write(os.path.join(folder, skip_name) + '\n')
            else:
                with open(self.skip_path, 'w') as f:
                    f.write(os.path.join(folder, skip_name) + '\n')



    def predict(self):
        imgs_path, label_path, class_num = self.load_data(self.data_path)
        for item, imgs in enumerate(tqdm(imgs_path)):  #volume
            image = nib.load(imgs).get_fdata()
            mask = nib.load(label_path[item]).get_fdata()
          
            #mount_preprocessed_sam/3d/semantic_seg/mr_de/Myops2020/labelsTr/myops_training_101.nii.gz
            #Evaluate-SAM/save_datasets/semantic_seg_3d_everything/mr_de/Myops2020/z/labelsTr/myops_training_101

            if image.shape[-1] > (self.image_size * 0.5) or self.dim == 'z': #判断长宽是或否适用
                vol_images, ori_label = self.process_img(image, mask, self.image_size, self.dim, class_num)
                #(slice, h, w, 3)  (slice, class, ori_h, ori_w)
                for i in range(vol_images.shape[0]): #slice

                    img_save_list = label_path[item].replace('.nii.gz',f'_{i}').split('/3d/semantic_seg/')
                    img_save_list[0] = self.save_path
                    img_save_path = '/'.join(img_save_list).replace('/labels', f"/{self.dim}/labels")

                    everything_ori_masks, predicted_iou = self.generate_mask(self.model, vol_images[i], ori_label[i], imgs, i)
                    if everything_ori_masks != None:
                        self.save_img(everything_ori_masks, predicted_iou, ori_label[i], img_save_path)
                        print(everything_ori_masks.shape, ori_label[i].shape)

                        
            else:
                print(f"This case: {imgs} does not meet the calculation criteria")



if __name__ == "__main__":
    
    args = parse_args()
    device_str = ','.join(str(i) for i in args.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{device_str}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = sam_model_registry[args.model_type](args.sam_checkpoint, args.image_size)
    
    print("Let's use", len(args.device_ids), "GPUs!")

    predictor = SamAutomaticMaskGenerator(  
        model.to(args.device),  #.module
        points_per_side = 32,
        points_per_batch = 64,
        pred_iou_thresh = 0.88,
        stability_score_thresh = 0.95,
        stability_score_offset = 1.0,
        box_nms_thresh = 0.7,
        crop_n_layers = 0,
        crop_nms_thresh = 0.7,
        crop_overlap_ratio = 512 / 1500,
        crop_n_points_downscale_factor = 1,
        point_grids = None,
        min_mask_region_area = 0,
        output_mode = "binary_mask",
        )

    eval_3d = everything_3d(args, predictor)
    eval_3d.predict()
