from segment_anything import sam_model_registry, SamPredictor
from data_load import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
from eval_utils import *
import argparse
import torch.nn as nn
import torch.nn.functional as F




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--data_path", type=str, default= 'mount_preprocessed_sam/2d/semantic_seg/dermoscopy/isic2017_task1/', help="eval data path")  #'mount_preprocessed_sam/2d/semantic_seg/endoscopy/EAD19/'
    parser.add_argument("--data_mode", type=str, default='val', help="eval train or test data")
    parser.add_argument("--metrics", nargs='+', default=['acc', 'iou', 'dice', 'sens', 'spec'], help="metrics")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[2, 3, 4, 5], help="device_ids")
    parser.add_argument("--model_type", type=str, default="vit_h", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="SAM-Med/pretrain_model/sam_vit_h.pth", help="sam checkpoint")
    parser.add_argument("--include_prompt_point", type=bool, default=False, help="need point prompt")
    parser.add_argument("--num_point", type=int, default=4, help="point or point number")
    parser.add_argument("--include_prompt_box", type=bool, default=True, help="need boxes prompt")
    parser.add_argument("--num_boxes", type=int, default=1, help="boxes or boxes number")
    parser.add_argument("--multimask_output", type=bool, default=True, help="multimask output")
    parser.add_argument("--save_path", type=str, default='Evaluate-SAM/save_datasets/2d/dermoscopy/isic2017_task1/', help="save data path")
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



def evaluate_batch_images(args, model):
    """
    evaluate the batch of images using a given model to produce output masks and iou predictions.

    args:
    - model: an instance of a pytorch model.
    - device: a string specifying the device on which to run the model (either 'cuda' or 'cpu').
    - data_path: a string specifying the path to the data directory containing the images to be evaluated.
    - batch_size: an integer specifying the number of images to process in each batch.
    - mode: a string indicating the type of the evaluation to perform ('train' or 'test'). default is 'train'.
    - include_prompt_point: a boolean indicating whether or not to include prompt points in the input. default is true.
    - include_prompt_box: a boolean indicating whether or not to include prompt boxes in the input. default is true.
    - num_point_boxes: an integer specifying the number of point boxes to include in the input. default is 1.

    returns:
    - best_score_masks: a tensor containing the mask with the highest iou score for each prompt point box in the dataset.
    - best_overlap_masks: a tensor containing the mask with the highest overlap score for each prompt point box in the dataset.
    - labels: a tensor containing the ground truth label for each image in the dataset.
    """

    
    dataset = Data_Loader(data_path=args.data_path,
                          image_size=args.image_size,
                          mode=args.data_mode,
                          prompt_point=args.include_prompt_point,
                          prompt_box=args.include_prompt_box,
                          num_boxes=args.num_boxes,
                          num_point=args.num_point
                          )

    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=48, collate_fn=collate_wrapper)
    progress_bar = tqdm(train_loader)


    save_path = os.path.join(args.save_path, "{}_{}_{}".format(
        args.image_size,
        "boxes" if args.include_prompt_box else "points",
        args.num_boxes if args.include_prompt_box else args.num_point
    ))

    txt_path = save_path + "/{}_{}_{}.txt".format(
        args.image_size,
        "boxes" if args.include_prompt_box else "points",
        args.num_boxes if args.include_prompt_box else args.num_point
    )

    os.makedirs(save_path, exist_ok=True)
    loggers = get_logger(txt_path)

    print('image number:', len(dataset))
    print("save_path:", save_path)
    l = len(train_loader)

    score_metric = [0] * len(args.metrics)
    overlap_metric =[0] * len(args.metrics)
    score_metrics, overlap_metrics = {}, {}

    for batch_input, ori_mask in progress_bar:
        batch_image = batch_input['image']         #B, Class, H, W, 3

        for i in range(batch_image.shape[0]):     #1, class, H, W, 3
            class_mask, class_score = [], []
            for j in range(batch_image.shape[1]):
                image = batch_image[i][j].astype(np.uint8)
                model.set_image(image)

                if args.include_prompt_point:
                    point_coords = batch_input['point_coords'][i][j]
                    point_labels = batch_input['point_labels'][i][j]
                else:
                    point_coords, point_labels = None, None

                if args.include_prompt_box:
                    if args.num_boxes > 1:
                        box = torch.as_tensor(batch_input['boxes'][i][j]).to(args.device)
                        box = model.transform.apply_boxes_torch(box, image.shape[:2])
                    else:
                        box = batch_input['boxes'][i][j][0]
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
                scores = torch.as_tensor(scores, dtype=torch.float)

                class_mask.append(masks)
                class_score.append(scores)

            class_out_mask = torch.stack(class_mask, dim=0)  #class, 3, 1, H, W
            class_out_iou = torch.stack(class_score, dim=0)  #class, 3, 1

            label = ori_mask[i]     #3, H, W
            # import pdb; pdb.set_trace()

            select_labels, class_idex = select_label(label)  #实际class, H, W  ——> 2, H, W
   
            select_class_outmask = class_out_mask[class_idex, ...]
            select_class_outiou = class_out_iou[class_idex, ...]

            best_iouscore_masks = select_mask_with_highest_iouscore(select_class_outmask, select_class_outiou)
            best_overlap_masks, overlap_score = select_mask_with_highest_overlap(select_class_outmask, select_labels)  #[class, 1, 1024, 1024], [class, n, 1]
       
            origin_size = batch_input['original_size']


            resize_scores, resize_overlaps, resize_labels = save_img(best_iouscore_masks, 
                                                                    best_overlap_masks, 
                                                                    select_labels, 
                                                                    save_path, 
                                                                    batch_input['name'][i], 
                                                                    class_idex, 
                                                                    origin_size[i]
                                                                    )  #class, 1, H, W

            class_score_metrics_ = SegMetrics(resize_scores, resize_labels, args.metrics)  #每张图片计算类别的 平均iou 和 dice
            class_overlap_metrics_ = SegMetrics(resize_overlaps, resize_labels, args.metrics) #每张图片计算类别的 平均iou 和 dice
            loggers.info(f"{batch_input['name'][i]} :\n iou score metrics: {class_score_metrics_}, overlap metrics: {class_overlap_metrics_}")
            
            for i in range(len(args.metrics)):
                score_metric[i] += class_score_metrics_[i]
                overlap_metric[i] += class_overlap_metrics_[i]

    for i, metr in enumerate(args.metrics):
        score_metrics[metr] = score_metric[i] / len(dataset)
        overlap_metrics[metr] = overlap_metric[i] / len(dataset)

    for key in score_metrics:
        score_metrics[key] = '{:.4f}'.format(score_metrics[key])
        overlap_metrics[key] = '{:.4f}'.format(overlap_metrics[key])

    loggers.info(f"get metrics on masks through iou score:  {score_metrics}")
    loggers.info(f"get metrics on masks through overlap:  {overlap_metrics}")

    return



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
