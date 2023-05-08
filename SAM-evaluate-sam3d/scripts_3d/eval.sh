#!/bin/bash

dataset=$1
modality=$2
device_id=$3
include_prompt_point=$4
num_point=$5
include_prompt_box=$6
num_boxes=$7

CUDA_VISIBLE_DEVICES=$device_id python evaluate3d.py --data_path /nvme/yejin/data/mount-medical_preprocessed/3d/semantic_seg/$modality/$dataset/ --sam_checkpoint /nvme/yejin/pretrain_model/sam_vit_h_4b8939.pth --save_path save_datasets/3d/semantic_seg/$modality/$dataset/ --device_ids $device_id --include_prompt_point $include_prompt_point --num_point $num_point --include_prompt_box $include_prompt_box --num_boxes $num_boxes
