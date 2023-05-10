import os
import glob
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import nibabel as nib
from data_load import random_point_sampling, get_box
import json


def copy_image(image_matrix, num_class):
    out_matrix = np.tile(image_matrix[:,np.newaxis,:,:,:], (1, num_class, 1, 1, 1))
    return out_matrix

class Data_Loader(Dataset):
    def __init__(self, data_path,
                 image_size = None,
                 requires_name = True,
                 dim='x',
                 prompt_point=True,
                 prompt_box=True,
                 num_boxes = 1,
                 num_point = 0):

        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.data_path = data_path
        if os.path.exists(os.path.join(data_path, 'labelsTs')):
            self.imgs_path = glob.glob(os.path.join(data_path, 'imagesTs/*.nii.gz'))
            self.label_path = glob.glob(os.path.join(data_path, 'labelsTs/*.nii.gz'))
        else:
            self.imgs_path = glob.glob(os.path.join(data_path, 'imagesTr/*.nii.gz'))
            self.label_path = glob.glob(os.path.join(data_path, 'labelsTr/*.nii.gz'))

        with open(os.path.join(data_path, 'dataset.json'), 'r') as f:
            data = json.load(f)
        self.num_class = len(data['labels'])

        self.imgs_path.sort()
        self.label_path.sort()

        self.requires_name = requires_name
        self.boxes_num = num_boxes
        self.point_num = num_point
        self.prompt_point = prompt_point
        self.prompt_box = prompt_box
        self.dim = dim


    def __getitem__(self, index):
        image_input = {}
        # 根据index读取图片
        image = nib.load(self.imgs_path[index]).get_fdata()
        mask = nib.load(self.label_path[index]).get_fdata()

        if image.shape[-1] < (self.image_size * 0.5) and self.dim != 'z':
            self.dim = 'z'
            print("We want the number of 'z' axes to be at least larger than (image size: %s * 0.5)"%self.image_size)
            print('Can only be tested on the "z" axis! image size:', image.shape)

        unique_values = np.unique(mask)
        if len(unique_values) == 1 and unique_values[0] == 0:
            image_input["image"] = np.array([0])
            image_input["label"] = np.array([0])
            image_input['ori_label'] = np.array([0])
            image_input["point_coords"] = np.array([0])
            image_input["point_labels"] = np.array([0])
            image_input["boxes"] = np.array([0])
            image_input["dim"] = self.dim
            image_input["zero_mask"] = np.array([0])
            image_input["index"] = np.array([0])
            if self.requires_name:
                image_input["name"] = self.imgs_path[index].split('/')[-1]
            return image_input

        target_size = (self.image_size, self.image_size)

        if self.dim == 'z':
            origin_slice = mask.shape[2]
            nonzero_slices = np.where(np.any(mask, axis=(0, 1)))[0]
            select_image = image[:, :, nonzero_slices]
            select_mask = mask[:, :, nonzero_slices]

        else:
            if self.dim == 'x':
                origin_slice = mask.shape[0]
                nonzero_slices = np.where(np.any(mask, axis=(1, 2)))[0]
                select_image = image[nonzero_slices, :, :].transpose(1,2,0)
                select_mask = mask[nonzero_slices, :, :].transpose(1,2,0)

            else:
                origin_slice = mask.shape[1]
                nonzero_slices = np.where(np.any(mask, axis=(0, 2)))[0]
                select_image = image[:, nonzero_slices, :].transpose(0,2,1)
                select_mask = mask[:,nonzero_slices, :].transpose(0,2,1)

        class_num = self.num_class
        max_pixel = select_image.max()
        min_pixel = select_image.min()
        select_image = (255 * (select_image - min_pixel) / (max_pixel - min_pixel)).astype(np.uint8)
  
        resized_image = cv2.resize(select_image, target_size, cv2.INTER_NEAREST)  #resize (H,W,1) -> (H,W)
        resized_mask = cv2.resize(select_mask, target_size, cv2.INTER_NEAREST).astype(np.int16) #resize (H,W,1) -> (H,W)
        ori_mask = select_mask.astype(np.int16)

        if len(resized_image.shape) == 2:
                resized_image = resized_image[:,:,np.newaxis]
        if len(resized_mask.shape) == 2:
                resized_mask = resized_mask[:,:,np.newaxis]

        volume_image = []
        volume_mask = []
        volume_ori_mask = []
        for i in range(resized_image.shape[-1]):
            volume_image.append(np.repeat(resized_image[...,i:i+1], repeats=3, axis=-1))
            volume_mask.append(resized_mask[...,i])
            volume_ori_mask.append(ori_mask[...,i])

        volume_images = np.stack(volume_image, axis=0)
        volume_masks = np.stack(volume_mask, axis=0)
        volume_ori_masks = np.stack(volume_ori_mask, axis=0)

        if len(volume_images.shape)<4:
            volume_images = np.expand_dims(volume_images, axis=0)

        if len(volume_masks.shape)<3:
            volume_masks = np.expand_dims(volume_masks, axis=0)

        if len(volume_ori_masks.shape)<3:
            volume_ori_masks = np.expand_dims(volume_ori_masks, axis=0)

        copy_img = copy_image(volume_images, class_num - 1)
        image_input["image"] = copy_img

        if class_num == 2:
            label = np.expand_dims(volume_masks, axis=1)
            ori_label = np.expand_dims(volume_ori_masks, axis=1)
        else:
            eye = np.eye(class_num, dtype=volume_masks.dtype)
            label = eye[volume_masks].transpose(0, 3, 1, 2)[:,1:,...]
            ori_label = eye[volume_ori_masks].transpose(0, 3, 1, 2)[:,1:,...]

        image_input["label"] = label
        image_input['ori_label'] = ori_label
        S, C, H, W = image_input["ori_label"].shape
        zero_mask = torch.zeros((origin_slice, C, H, W), dtype=torch.int)


        #image (slice, class-1, H, W, 3)
        #label (slice, class-1, H, W)
        if self.prompt_point:
            points, point_labels = [], []
            for i in range(label.shape[0]): #切片
                point, point_label = random_point_sampling(label[i], point_num=self.point_num)
                points.append(point)
                point_labels.append(point_label)

            points_ = torch.stack(points, dim=0)
            point_labels_ = torch.stack(point_labels, dim=0)
            image_input["point_coords"] = points_
            image_input["point_labels"] = point_labels_

        if self.prompt_box:
            boxes_ = []
            for i in range(label.shape[0]): #切片
                box = get_box(label[i], num_classes=self.boxes_num)
                boxes_.append(box)
            boxes = torch.stack(boxes_, dim=0)
            image_input["boxes"] = boxes

        image_name = self.imgs_path[index].split('/')[-1]
        image_input["dim"] = self.dim
        image_input["zero_mask"] = zero_mask
        image_input["index"] = nonzero_slices

        if self.requires_name:
            image_input["name"] = image_name

            return image_input
        else:
            return image_input

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    train_dataset = Data_Loader('mount_preprocessed_sam/3d/semantic_seg/ct_mtt/ISLES2018/', image_size=224, prompt_point=False, prompt_box=False, dim='z',num_boxes = 0, num_point = 0)
    print("数据个数：", len(train_dataset))
    train_batch_sampler = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    for batched_image in (train_batch_sampler):
        print(batched_image['name'])
        print(batched_image['image'].shape)
        print('*'*20)