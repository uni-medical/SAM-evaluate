#!/bin/bash

dataset=$1
modality=$2
device_id=$3

CUDA_VISIBLE_DEVICES=$device_id python evaluate3d.py --data_path /nvme/yejin/data/mount-medical_preprocessed/3d/semantic_seg/$modality/$dataset/ --sam_checkpoint /nvme/yejin/pretrain_model/sam_vit_h_4b8939.pth --save_path save_datasets/3d/semantic_seg/$modality/$dataset/ --device_ids $device_id
