#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 25887 --use_env bare_train.py \
    --num_workers 16 --epochs 60 --batch_size 32 --lr 0.0001   \
    --data_root ./ln_data --split_root ./data \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --beit_model ./weights/beit3_large_indomain_patch16_224.pth \
    --output_dir ./work_dir/BARE/mixup_large \
    --sup_type full --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --imsize 224 --max_query_len 77 \
    --normalize_before --enable_adaptive_weights \
    --dataset mixup --use_contrastive_loss --use_rtcc_constrain_loss --use_mask_loss;

infer_pth='./work_dir/BARE/mixup_large/best_checkpoint.pth'
# unc
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set val \
    --output_dir ./work_dir/BARE/mixup_large_infer;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set testA \
    --output_dir ./work_dir/BARE/mixup_large_infer;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set testB \
    --output_dir ./work_dir/BARE/mixup_large_infer;

# unc+
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc+ \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set val \
    --output_dir ./work_dir/BARE/mixup_large_infer;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc+ \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set testA \
    --output_dir ./work_dir/BARE/mixup_large_infer;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset unc+ \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set testB \
    --output_dir ./work_dir/BARE/mixup_large_infer;

# gref_umd
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset gref_umd \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set val \
    --output_dir ./work_dir/BARE/mixup_large_infer;
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset gref_umd \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set test \
    --output_dir ./work_dir/BARE/mixup_large_infer;

# referit
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset referit \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set test \
    --output_dir ./work_dir/BARE/mixup_large_infer;

# flickr
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 28888 --use_env bare_eval.py \
    --num_workers 8 --batch_size 64  --dataset flickr \
    --vit_type large --vl_hidden_dim 1024 --vl_dim_feedforward 4096 \
    --imsize 224 --max_query_len 77 --normalize_before --enable_adaptive_weights --use_mask_loss  \
    --data_root ./ln_data --split_root ./data \
    --eval_model $infer_pth --eval_set test \
    --output_dir ./work_dir/BARE/mixup_large_infer;
