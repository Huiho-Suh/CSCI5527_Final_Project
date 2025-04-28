```
bash
torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/test \
    --num_epochs 1  \
    --batch_size 64 \
    --img_scale_factor 0.5 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 

-- CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/test \
    --num_epochs 1  \
    --batch_size 64 \
    --img_scale_factor 0.5 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 

```
scale, patch
0.5, 16
0.25, 8