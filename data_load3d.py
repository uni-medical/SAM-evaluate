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
                 num_boxes = 2,
                 num_point = 2):

        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.data_path = data_path
        if os.path.exists(os.path.join(data_path, 'labelsTs')):
            self.imgs_path = glob.glob(os.path.join(data_path, 'imagesTs/*.nii.gz'))
            self.label_path = glob.glob(os.path.join(data_path, 'labelsTs/*.nii.gz'))
        else:
            self.imgs_path = glob.glob(os.path.join(data_path, 'imagesTr/*.nii.gz'))
            self.label_path = glob.glob(os.path.join(data_path, 'labelsTr/*.nii.gz'))

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

        target_size = (self.image_size, self.image_size)
        if self.dim == 'z':
            origin_slice = mask.shape[2]
            nonzero_slices = np.where(np.any(mask, axis=(0, 1)))[0]
            select_image = image[:, :, nonzero_slices]
            selcet_mask = mask[:, :, nonzero_slices]

        else:
            if self.dim == 'x':
                origin_slice = mask.shape[0]
                nonzero_slices = np.where(np.any(mask, axis=(1, 2)))[0]
                select_image = image[nonzero_slices, :, :].transpose(1,2,0)
                selcet_mask = mask[nonzero_slices, :, :].transpose(1,2,0)

            else:
                origin_slice = mask.shape[1]
                nonzero_slices = np.where(np.any(mask, axis=(0, 2)))[0]
                select_image = image[:, nonzero_slices, :].transpose(0,2,1)
                selcet_mask = mask[:,nonzero_slices, :].transpose(0,2,1)


        class_num = len(np.unique(selcet_mask))

        resized_image = cv2.resize(select_image, target_size, cv2.INTER_NEAREST)
        resized_mask = cv2.resize(selcet_mask, target_size, cv2.INTER_NEAREST).astype(np.int)

        volume_image = []
        volume_mask = []
        for i in range(resized_image.shape[-1]):
            volume_image.append(np.repeat(resized_image[...,i:i+1], repeats=3, axis=-1))
            volume_mask.append(resized_mask[...,i])

        volume_images = np.stack(volume_image, axis=0)
        volume_masks = np.stack(volume_mask, axis=0)

        if len(volume_images.shape)<4:
            volume_images = np.expand_dims(volume_images, axis=0)

        if len(volume_masks.shape)<3:
            volume_masks = np.expand_dims(volume_masks, axis=0)

        copy_img = copy_image(volume_images, class_num - 1)
        image_input["image"] = copy_img

        if class_num == 2:
            label = np.expand_dims(volume_masks, axis=1)
        else:
            eye = np.eye(class_num, dtype=volume_masks.dtype)
            label = eye[volume_masks].transpose(0, 3, 1, 2)[:,1:,...]

        image_input["label"] = label
        S, C, H, W = image_input["label"].shape
        zero_mask = torch.zeros((origin_slice, C, H, W), dtype=torch.int)

        if self.prompt_point and class_num == 2:
            points, point_labels = [], []
            for i in range(label.shape[0]):
                point, point_label = random_point_sampling(label[i][0], point_num=self.point_num)
                points.append(point)
                point_labels.append(point_label)
            points = torch.stack(points, dim=0)
            point_labels = torch.stack(point_labels, dim=0)
            image_input["point_coords"] = points.unsqueeze(1)
            image_input["point_labels"] = point_labels.unsqueeze(1)

        #image (slice, class-1, H, W, 3)
        #label (slice, class-1, H, W)
        if self.prompt_point and class_num > 2:
            points, point_labels = [], []
            for i in range(label.shape[0]): #切片
                class_point, class_point_label = [], []
                for j in range(label.shape[1]): #类别
                    point, point_label = random_point_sampling(label[i][j], point_num=self.point_num)
                    class_point.append(point)
                    class_point_label.append(point_label)

                points.append(torch.stack(class_point, dim=0))
                point_labels.append(torch.stack(class_point_label, dim=0))

            points_ = torch.stack(points, dim=0)
            point_labels_ = torch.stack(point_labels, dim=0)

            image_input["point_coords"] = points_
            image_input["point_labels"] = point_labels_


        if self.prompt_box and class_num == 2:
            boxes = []
            for i in range(label.shape[0]):
                box = get_box(label[i][0], num_classes=self.boxes_num)
                boxes.append(box)
            boxes = torch.stack(boxes, dim=0)
            image_input["boxes"] = boxes.unsqueeze(1)

        if self.prompt_point and class_num > 2:
            boxes_ = []
            for i in range(label.shape[0]): #切片
                class_box = []
                for j in range(label.shape[1]): #类别
                    box = get_box(label[i][0], num_classes=self.boxes_num)
                    class_box.append(box)
                boxes_.append(torch.stack(class_box, dim=0))
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
    train_dataset = Data_Loader("datasets/autoPET/", image_size=256, prompt_point=True, prompt_box=True, dim='x',)
    print("数据个数：", len(train_dataset))
    train_batch_sampler = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    for batched_image in (train_batch_sampler):
        print('*'*20)
        print(batched_image['image'].shape)
        print(batched_image['label'].shape)
        print(batched_image.get('point_coords', None).shape)
        print(batched_image.get('point_labels', None).shape)
        print(batched_image.get('boxes', None).shape)
        print(batched_image['dim'])
