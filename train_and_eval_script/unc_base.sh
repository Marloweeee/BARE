#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28887 --use_env bare_train.py \
    --num_workers 16 --epochs 60 --batch_size 32 --lr 0.0001   \
    --data_root ./ln_data --split_root ./data \
    --beit_model ./weights/beit3_base_indomain_patch16_224.pth \
    --output_dir ./work_dir/BARE/unc_base \
    --sup_type full --lr_scheduler cosine --aug_crop --aug_scale --aug_translate   --vl_hidden_dim 768 --vl_dim_feedforward 3072  --imsize 224 --max_query_len 77 \
    --normalize_before --enable_adaptive_weights --dataset unc --use_contrastive_loss  --use_rtcc_constrain_loss --use_mask_loss;
    # --resume ./work_dir/BARE/unc_base/checkpoint.pth

# Infer val
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc --vl_hidden_dim 768 --vl_dim_feedforward 3072 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model ./work_dir/BARE/unc_base/best_checkpoint.pth --eval_set val \
    --output_dir ./work_dir/BARE/unc_base_infer;

# Infer testA
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc --vl_hidden_dim 768 --vl_dim_feedforward 3072 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model ./work_dir/BARE/unc_base/best_checkpoint.pth --eval_set testA \
    --output_dir ./work_dir/BARE/unc_base_infer;

# Infer testB
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc --vl_hidden_dim 768 --vl_dim_feedforward 3072 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model ./work_dir/BARE/unc_base/best_checkpoint.pth --eval_set testB \
    --output_dir ./work_dir/BARE/unc_base_infer;
