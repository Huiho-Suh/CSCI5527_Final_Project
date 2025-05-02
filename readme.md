# 1. Requirements
## (1) Create a Conda Environment
```
conda create env -n csci5527 -f environment.yml
conda activate csci5527
```
## (2) Install Torch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# 2. Training and Evaluating
For training, run the following script with appropriate arguments. \
For evaluating, add `--eval` to the existing command.
```

torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_2_100 \
    --num_epochs 1  \
    --model_type CNN \
    --batch_size 32 \
    --img_scale_factor 2 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 


# To specify the GPU to be trainied:
-- CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_2_100 \
    --num_epochs 1  \
    --batch_size 32 \
    --img_scale_factor 2 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 

    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_2_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 2 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0

    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_2_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 2 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 --eval

    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_4_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 4 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0

    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_4_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 4 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 --eval


    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_8_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 8 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0

    torchrun --nproc_per_node=2 train.py \
    --task_name aloha_sim_insertion_human_image \
    --ckpt_dir ckpt/insertion_sr_8_Transformer \
    --num_epochs 100  \
    --model_type Transformer \
    --batch_size 16 \
    --img_scale_factor 8 \
    --embed_dim 768 \
    --dim_feedforward 2048 \
    --weight_decay 1e-4 \
    --lr 1e-3 \
    --seed 0 --eval

```


### Options
1. nproc_per_node: Number of GPUs available 
2. task_name: Name of the robot task. For this project, aloha_sim_insertion_human_image
3. model_type: Type of the model that the VAE is based on. (Either CNN or Transformer)
4. ckpt_dir: Directory of the ckpt.
5. img_scale_factor: Scaling factor for both H & W.