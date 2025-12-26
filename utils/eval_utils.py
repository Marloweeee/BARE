import math

import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy
from utils.misc import mdetr_interpolate


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def statistic(text_list, iou):
    statistic_diff_length_acc = False
    if statistic_diff_length_acc:  # only can be used in one GPU, used for result comparison.
        # calculate text length, statistics
        assert len(text_list) == iou.shape[0]
        count_for_len_in_1_to_5 = [0, 0]
        count_for_len_in_6_to_7 = [0, 0]
        count_for_len_in_8_to_10 = [0, 0]
        count_for_len_in_11_plus = [0, 0]
        for i in range(len(text_list)):
            len_i = len(text_list[i].split(" "))
            iou_i = iou[i]
            if (len_i >= 1) and (len_i <= 5):
                count_for_len_in_1_to_5[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_1_to_5[0] += 1
            elif (len_i >= 6) and (len_i <= 7):
                count_for_len_in_6_to_7[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_6_to_7[0] += 1
            elif (len_i >= 8) and (len_i <= 10):
                count_for_len_in_8_to_10[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_8_to_10[0] += 1
            elif (len_i >= 11):
                count_for_len_in_11_plus[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_11_plus[0] += 1

        print("acc in length  1-5: ", count_for_len_in_1_to_5, ", ",
              count_for_len_in_1_to_5[0] / count_for_len_in_1_to_5[1])
        print("acc in length  6-7: ", count_for_len_in_6_to_7, ", ",
              count_for_len_in_6_to_7[0] / count_for_len_in_6_to_7[1])
        print("acc in length 8-10: ", count_for_len_in_8_to_10, ", ",
              count_for_len_in_8_to_10[0] / count_for_len_in_8_to_10[1])
        print("acc in length  11+: ", count_for_len_in_11_plus, ", ",
              count_for_len_in_11_plus[0] / count_for_len_in_11_plus[1])



# IoU calculation for validation, This IOU can only be implemented in the single GPU card environment.
def IoU(pred, gt):
    # pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def IoU_v2(pred, gt):
    # [B h*w]
    intersection = torch.sum(torch.mul(pred, gt), dim=1)
    union = torch.sum(torch.add(pred, gt), dim=-1) - intersection
    iou_list = torch.div(intersection, union)

    return iou_list, intersection, union


def trans_vg_eval_val(args, pred_boxes, gt_boxes, pred_mask, tgt_mask):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    if args.use_mask_loss:
        # ReLU needs to be applied before interpolation, otherwise it will affect the interpolation effect.
        pred_mask = (pred_mask.sigmoid() - torch.tensor(0.5)).relu()
        src_mask = mdetr_interpolate(pred_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1).bool().float()
        tgt_mask = tgt_mask.flatten(1).float()  # from [B 1 h w] to [B h*w]
        mask_iou_list, I, U = IoU_v2(src_mask, tgt_mask)
    else:
        mask_iou_list = torch.tensor([])

    return iou, accu, mask_iou_list


def trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_mask, tgt_mask):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    # XLH 新增
    if args.use_mask_loss:
        # ReLU needs to be applied before interpolation, otherwise it will affect the interpolation effect.
        pred_mask = (pred_mask.sigmoid() - torch.tensor(0.5)).relu()
        # patch_num = int(math.sqrt(pred_mask.shape[-1]))
        # pred_mask = pred_mask.reshape(pred_mask.shape[0], 1, patch_num, patch_num)
        src_mask = mdetr_interpolate(pred_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1).bool().float()
        tgt_mask = tgt_mask.flatten(1).bool().float()  # from [B 1 h w] to [B h*w]
        mask_iou_list, I, U = IoU_v2(src_mask, tgt_mask)
    else:
        mask_iou_list = torch.tensor([])

    return accu_num, iou, mask_iou_list


def trans_vg_eval_test_iou(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num, iou
