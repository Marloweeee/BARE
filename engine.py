# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import xywh2xyxy
import numpy as np


# TODO: 训练核心代码
def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0):
    # 设置模型在训练模式
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # TODO: 这是从 data_loader.py 的 TransVGdataset 的 __getitem__()中获取的
        img_data, text_data, target, obj_mask = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)

        # model forward core-computer
        output, seg_mask, logits_per_image, visu_token_similarity = model(
            img_data, text_data)

        # 获取去偏门控的 alpha 值（用于监控）
        # 处理 DDP 包装的情况
        base_model = model.module if hasattr(model, 'module') else model
        gate_info = base_model.get_last_gate_info()
        gate_alpha = gate_info['alpha'].item(
        ) if gate_info is not None else 0.0

        loss_dict = loss_utils.bare_loss(
            args, output, target, seg_mask, obj_mask, logits_per_image, visu_token_similarity)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # The default value of max_norm is 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(gate_alpha=gate_alpha)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)

        pred_boxes, seg_mask, logits_per_image, visu_sim = model(
            img_data, text_data)
        # accu,miou=eval_utils.trans_vg_eval_test_iou(pred_boxes, target)
        miou, accu, mask_iou_list = eval_utils.trans_vg_eval_val(
            args, pred_boxes, target, seg_mask, tgt_mask)

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)
        metric_logger.update_v2(
            'mask seg miou', torch.mean(mask_iou_list), batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    text_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU

        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        output, seg_mask, logits_per_image, visu_token_similarity = model(
            img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

        # for text_i in text_data:
        #     text_list.append(text_i)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)

    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list = eval_utils.trans_vg_eval_test(
        args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        # It is work only used for referring image segmentation task and enable use args.use_seg_mask
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print("accuracy2: ", accuracy)
    if args.use_mask_loss:
        # It is work only used for referring image segmentation task and enable use args.use_seg_mask
        miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        print("segmentation miou: ", miou)

    return accuracy, miou


@torch.no_grad()
def evaluate_ori(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, obj_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        output, _, _, _, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy
