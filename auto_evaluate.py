from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from data_load import Data_Loader, get_origin_size
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from eval_utils import SegMetrics, get_logger, select_label
import argparse
import torch.nn as nn
import os
import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torchvision

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10, help="train batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--data_path", type=str, default='mount_preprocessed_sam/2d/semantic_seg/mr/UW-Madison/', help="eval data path")
    parser.add_argument("--data_mode", type=str, default='val', help="eval train or test data")
    parser.add_argument("--metrics", nargs='+', default=['acc', 'iou', 'dice', 'sens', 'spec'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_h", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="Evaluate-SAM/pretrain_model/sam_vit_h_4b8939.pth", help="sam checkpoint")
    parser.add_argument("--save_path", type=str, default='Evaluate-SAM/save_datasets/2d/UW-Madison/', help="save data path")
    args = parser.parse_args()

    return args


def save_img(predict, label, save_path, mask_name, class_idex, origin_size):

    predict = predict.cpu().numpy()
    label = label.cpu().numpy()
    class_idex = class_idex.cpu().numpy()
    N, H, W = predict.shape

    save_pred_path = os.path.join(save_path, 'predict_masks')
    save_label_path = os.path.join(save_path, 'masks')

    os.makedirs(save_pred_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    trans_totensor = torchvision.transforms.ToTensor()
    resize_pred, resize_label = [], []
    for i in range(N):
        pred_img = Image.fromarray(np.uint8(predict[i] * 255))
        label_img = Image.fromarray(np.uint8(label[i] * 255))

        save_name = mask_name.split('.')[0] + '_' + str(class_idex[i]+1).zfill(3) + '.png'

        resized_pred_img = pred_img.resize(origin_size, resample=Image.NEAREST)
        resized_label_img = label_img.resize(origin_size, resample=Image.NEAREST)

        resize_pred.append(trans_totensor(resized_pred_img))
        resize_label.append(trans_totensor(resized_label_img))

        resized_pred_img.save(os.path.join(save_pred_path, save_name))
        resized_label_img.save(os.path.join(save_label_path, save_name))

    resize_preds = torch.stack(resize_pred, dim=0).unsqueeze(1) / 255.
    resize_labels = torch.stack(resize_label, dim=0).unsqueeze(1) / 255.
    return resize_preds, resize_labels


def label_mask_overlap(label, mask):
    N = label.shape[0]
    B = mask.shape[0]
    # 计算每个通道的重叠率
    overlap = torch.zeros(B, N)        
    for i in range(B):
        for j in range(N):
            overlap[i][j] = torch.sum(torch.logical_and(label[j], mask[i])) / torch.sum(torch.logical_or(label[j], mask[i]))
    # 取出前N个最大的重叠率的索引
    indices = torch.argmax(overlap, dim=0)

    # 提取pred中的对应通道
    output = torch.index_select(mask, 0, indices)

    return output

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def evaluate_batch_images(args, model):

    dataset = Data_Loader(
                        args.data_path, 
                        image_size = args.image_size, 
                        requires_name = True, 
                        mode = args.data_mode, 
                        prompt_point=False, 
                        prompt_box=False, 
                        num_boxes = 0, 
                        num_point = 0,
                    )

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=48)
    progress_bar = tqdm(train_loader)

    save_path = os.path.join(args.save_path, "auto_predict")
    txt_path = save_path + "/{}.txt".format("auto_predict")

    os.makedirs(save_path, exist_ok=True)
    loggers = get_logger(txt_path)

    print('image number:', len(dataset))
    print("save_path:", save_path)


    class_metrics = [0] * len(args.metrics)
    total_metrics = {}

    for batch_input in progress_bar:
        batch_image = batch_input['image']
        for i in range(batch_image.shape[0]):  #class,H,W,3
            for j in range(batch_image.shape[1]):  #1,H,W,3  
                image = batch_image[i][j].cpu().numpy().astype(np.uint8)
                masks = model.generate(image)

                class_mask = [x['segmentation'] for x in masks]  #一个class
            
            class_labels = batch_input['label'][i].type(torch.int)       #class, h, w 
            select_labels, class_idex = select_label(class_labels)

            class_masks = torch.as_tensor(np.stack(class_mask, axis=0), dtype=torch.int)
            select_masks = label_mask_overlap(select_labels, class_masks)                    #class, h, w
            
            origin_size = get_origin_size(batch_input['original_size'])
            resize_preds, resize_labels = save_img(select_masks, 
                                                   select_labels, 
                                                   save_path, 
                                                   batch_input['name'][i], 
                                                   class_idex,
                                                   origin_size[i][::-1]
                                                   )

            class_metrics_ = SegMetrics(resize_preds, resize_labels, args.metrics)  #每张图片计算类别的 平均iou 和 dice
            loggers.info(f"{batch_input['name'][i]} metrics:\n {class_metrics_}")
            for i in range(len(args.metrics)):
                class_metrics[i] += class_metrics_[i]

    for i, metr in enumerate(args.metrics):
        total_metrics[metr] = class_metrics[i] / len(dataset)

    loggers.info(f"The mean metrics of {save_path.split('/')[-2]} dataset is: {total_metrics}")

    return





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    args.device = device
    model = sam_model_registry[args.model_type](args.sam_checkpoint)
    if len(args.device_ids) > 1:
        print("Let's use", len(args.device_ids), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.device_ids)
        predictor = SamAutomaticMaskGenerator(model.module.to(args.device))
    else:
        predictor = SamAutomaticMaskGenerator(model.to(args.device))

    evaluate_batch_images(args, predictor)
