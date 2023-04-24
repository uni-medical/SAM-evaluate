import os
import glob
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random

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

def draw_boxes(img, boxes):
    boxes = boxes.astype(np.int)
    img = np.expand_dims(img, -1).repeat(3, axis=-1).astype(np.uint8)
    img_copy = np.copy(img)
    img_copy = np.ascontiguousarray(img_copy)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    plt.imshow(img_copy)
    plt.show()


def show_torch_img(image, label):
    imgs, lab = image.cpu().numpy().transpose(1, 2, 0), label.cpu().numpy().transpose(1, 2, 0) * 255.
    plt.imshow(imgs[...,::-1])
    plt.show()
    plt.imshow(lab)
    plt.show()


def form_mask_get_box(mask, std = 0.1, max_pixel = 20):
    mask = mask.numpy() * 255.
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    stats = stats[stats[:,4].argsort()]
    boxs = stats[:-1]
    max_weight = 0
    x0, y0, x1, y1 = 0, 0, 0, 0
    if len(boxs) > 1:
        for box in boxs:
            x0, y0 = box[0], box[1]
            x1, y1 = box[0] + box[2], box[1] + box[3]  # start_point, end_point = (x0, y0), (x1, y1)
            # 计算长方体边框的宽度和高度
            if abs(x1 - x0) > max_weight:
                max_weight = abs(x1 - x0)
                x0, y0, x1, y1 = x0, y0, x1, y1
    else:
        box = boxs[0]
        x0, y0 = box[0], box[1]
        x1, y1 = box[0] + box[2],box[1] + box[3]   #start_point, end_point = (x0, y0), (x1, y1)
    # 计算长方体边框的宽度和高度
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    # 计算标准差和最大噪声值
    noise_std = min(width, height) * std
    max_noise = min(max_pixel, int(noise_std * 5))
    # 在每个坐标上添加随机噪音,应用噪声到左上角坐标
    noise_x = np.random.randint(-max_noise, max_noise)
    noise_y = np.random.randint(-max_noise, max_noise)
    x0 = x0 + noise_x
    y0 = y0 + noise_y
    # 应用噪声到右下角坐标
    noise_x = np.random.randint(-max_noise, max_noise)
    noise_y = np.random.randint(-max_noise, max_noise)
    x1 = x1 + noise_x
    y1 = y1 + noise_y
    return torch.as_tensor(np.array([(x0, y0, x1, y1)]), dtype=torch.float)

def get_box(mask, num_classes=4, std = 0.1, max_pixel = 20):
    # 标记图像并获取区域属性
    mask = np.array(mask)
    label_img = label(mask)
    regions = regionprops(label_img)
    # 迭代所有区域并获取边界框坐标
    boxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        boxes.append((minc, minr, maxc, maxr))  # 坐标存储为 (x_min, y_min, x_max, y_max) 的形式
    # 如果生成的边界框数量大于类别数，则按区域面积排序并选择前n个区域
    if len(boxes) > num_classes:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:num_classes]
        boxes = []
        for region in sorted_regions:
            minr, minc, maxr, maxc = region.bbox
            boxes.append((minc, minr, maxc, maxr))
    elif len(boxes) < num_classes:
        if len(boxes) == 0:
            boxes.append([256, 256, 512, 512])
        num_duplicates = num_classes - len(boxes)
        for i in range(num_duplicates):
            boxes.append(random.choice(boxes))

    noise_boxes = []
    for box in boxes:
        x0, y0 = box[0], box[1]
        x1, y1 = box[2], box[3]   #start_point, end_point = (x0, y0), (x1, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        # 计算标准差和最大噪声值
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
        # 在每个坐标上添加随机噪音,应用噪声到左上角坐标
        if max_noise == 0:
            noise_x = 0
            noise_y = 0
        else:
            noise_x = np.random.randint(-max_noise, max_noise)
            noise_y = np.random.randint(-max_noise, max_noise)
        x0 = x0 + noise_x
        y0 = y0 + noise_y
        # 应用噪声到右下角坐标
        if max_noise == 0:
            noise_x = 0
            noise_y = 0
        else:
            noise_x = np.random.randint(-max_noise, max_noise)
            noise_y = np.random.randint(-max_noise, max_noise)
        x1 = x1 + noise_x
        y1 = y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))

    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(pred, gt, point_num = 9):
    # 计算错误区域
    pred, gt = pred.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1
    # error = np.logical_xor(pred, gt)

    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)
        # 随机选择9个点
        indices = np.argwhere(one_erroer == 1)
        selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=False)]
        selected_indices = selected_indices.reshape(-1, 2)
        # 对于每个选定的点，计算标签
        points = []
        labels = []
        for i in selected_indices:
            x = i[0]
            y = i[1]
            # 如果该点是假阴性，则将其标记为前景
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            # 如果该点是假阳性，则将其标记为前景
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 1
            # 否则将其标记为背景
            else:
                label = 0
            points.append((y, x))   #这里坐标相反
            labels.append(label)
        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def random_point_sampling(img, point_num = 2):
    img = np.array(img)

    # 获取黑/白色像素的坐标
    white_pixels = np.argwhere(img == 1)
    # 从中随机选择num_points个点
    if len(white_pixels) > 12:
        coords = white_pixels[np.random.choice(white_pixels.shape[0], point_num, replace=False)]
        coords[:, [0, 1]] = coords[:, [1, 0]]
        labels = np.ones(shape=(point_num))

    else:
        #随机选择点
        coords = []
        for i in range(point_num):
            x, y = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
            coords.append((x, y))
        labels = []
        for coord in coords:
            if img[coord[0], coord[1]] > 0:
                labels.append(1)
            else:
                labels.append(0)
        # print(np.array(coords).shape)
        # print(np.array(labels).shape)

    labels = torch.as_tensor(labels, dtype=torch.int)
    coords = torch.as_tensor(coords, dtype=torch.float)
    return coords, labels


def transforms(img_size):
    return A.Compose([
            A.Resize(img_size, img_size),
            # ToTensorV2(p=1.0),
        ], p=1.)


class Data_Loader(Dataset):
    def __init__(self, data_path, image_size = None, requires_name = True, mode = 'train', prompt_point=True, prompt_box=True, num_boxes = 1, num_point = 1):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.data_path = data_path

        self.img_list = [line.strip() +".png" for line in open(os.path.join(data_path, mode +".txt"), "r").readlines()]
        self.imgs_path = [os.path.join(data_path, "images", img) for img in self.img_list]
        self.label_path = [os.path.join(data_path, "masks", img) for img in self.img_list]
        #self.imgs_path = [f for f in glob.glob(os.path.join(data_path, 'images', mode, '*.*')) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]
        #self.label_path = [f for f in glob.glob(os.path.join(data_path, 'masks', mode, '*.*')) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]

        self.imgs_path.sort()
        self.label_path.sort()
        self.transforms = transforms(image_size)
        self.requires_name = requires_name
        self.boxes_num = num_boxes
        self.point_num = num_point
        self.prompt_point = prompt_point
        self.prompt_box = prompt_box

    def __getitem__(self, index):
        image_input = {}
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.label_path[index], 0)
        if mask.max() == 255:
            print("Mask is not a binary map")
            mask = mask // 255.

        # 读取训练图片和标签图片
        augments = self.transforms(image=image, mask=mask)
        # image, mask = augments['image'].transpose(2, 0, 1), augments['mask'].astype(np.int64)
        image, mask = augments['image'], augments['mask'].astype(np.int64)

        image_input["image"] = image
        image_input["label"] = np.expand_dims(mask, axis=0)

        if self.prompt_point:
            point, point_label = random_point_sampling(mask, point_num = self.point_num)
            # draw_boxes(mask.cpu().numpy()*255, box.cpu().numpy())
            image_input["point_coords"] = point
            image_input["point_labels"] = point_label

            # image_input["point_coords"] = point.unsqueeze(1)
            # image_input["point_labels"] = point_label.unsqueeze(1)

        if self.prompt_box:
            box = get_box(mask, num_classes=self.boxes_num)
            image_input["boxes"] = box

        image_name = self.imgs_path[index].split('/')[-1]

        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    train_dataset = Data_Loader("/home/chengjunlong//mount_preprocessed_sam/2d/semantic_seg/fundus_photography/drac2022_taska1/", image_size=1024, mode='train', prompt_point=True, prompt_box=False)
    print("数据个数：", len(train_dataset))
    train_batch_sampler = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=False)
    for batched_image in (train_batch_sampler):
        print(batched_image['image'].shape)
        print(batched_image['label'].shape)
        print(torch.unique(batched_image['label']))
        # print(batched_image.get('boxes', None))