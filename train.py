import torch
from torch import nn
import pickle
import os
import torch.nn as nn

from tqdm import tqdm
import logging
import argparse

# Distributed training
import torch.distributed as dist

# Mixed Pricision Training
from torch.amp import autocast

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity

from utils import set_seed, load_data, set_distributed_training, initialize_model_ddp, plot_history, save_examples
from utils import frechet_distance, get_beta_cyclical, inception_preprocess, compute_dict
from tqdm import tqdm

from torchvision.models import inception_v3
import gc


def main(args):
    # Set random seed
    set_seed(1)
    
    # Get config
    config = get_config(args)
    model_config = config['model_config']
    is_eval = config['eval']
    ckpt_dir = model_config['ckpt_dir']
    

    os.makedirs(ckpt_dir, exist_ok=True)
    
    ddp_config = set_distributed_training()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model, optimizer, scaler = initialize_model_ddp(model_config, ddp_config)
    
    # Load dataset
    train_dataloader, val_dataloader, norm_stats = load_data(config)
    
    # Evaluate model (inference mode)
    if is_eval:
        eval(model, model_config, val_dataloader, ddp_config)
        exit()
    
    # Train model
    train(model, model_config, optimizer, train_dataloader,
          val_dataloader, scaler, ddp_config)
    
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)
    
def train(model, model_config, optimizer, 
          train_dataloader,
          val_dataloader,
          scaler, ddp_config):
    torch.cuda.empty_cache()
    # Hyperparameters
    num_epochs = model_config['num_epochs']
    ckpt_dir = model_config['ckpt_dir']
    device = ddp_config['device']
    local_rank = ddp_config['local_rank']
    seed = model_config['seed']
    min_val_loss = float('inf')
    set_seed(seed)
    best_loss = float('inf')
    best_ckpt = None
    
    train_history = []
    validation_history = []
    
    # Validating
    if local_rank == 0:
        epoch_iter = tqdm(range(num_epochs), desc="Training")
        
    else:
        epoch_iter = range(num_epochs)
    
    for epoch in epoch_iter:
        
        val_dataloader.sampler.set_epoch(epoch) # Set epoch for distributed sampler
        # Validation
        total_loss = torch.zeros(3, device=device)
        total_count = torch.tensor(0, device=device)
        
        with torch.inference_mode():
            model.eval()
            # Cyclical KL Divergence Annealing
            cycle_length = 500
            iteration = 0
            for batch in val_dataloader:
                beta = get_beta_cyclical(iteration, cycle_length, min_beta=0.1, max_beta=1.0)
                batch = batch.to(device, non_blocking=True)
                recon_batch, mu, log_var = model(batch)
                bce, kl, loss = model.module.loss_function(batch, recon_batch, mu, log_var, beta)
            
                total_loss += torch.tensor((bce.detach(), kl.detach(), loss.detach()), device=device)
                total_count += batch.size(0)
                
                del batch, recon_batch, mu, log_var
                torch.cuda.empty_cache()
                
        # Sum up the losses from all GPUs
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_count, dst=0, op=dist.ReduceOp.SUM)

        
        if local_rank == 0:
            
            # avg_loss = total_loss.item() / total_count.item()
            avg_loss = total_loss / total_count
            validation_history.append({'bce_loss': avg_loss[0].item(), 'kl_loss': avg_loss[1].item(), 'total_loss': avg_loss[2].item()})
            # print(f"Validation Loss: {avg_loss[2].item():.4f}")
            
            if loss < best_loss:
                best_loss = loss
                ckpt = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': avg_loss[2],
                }
                best_ckpt = ckpt
                # print('best epoch', best_epoch)
                torch.save(best_ckpt, os.path.join(ckpt_dir, 'best_ckpt.pth'))
                # print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
                
        
        # Training
        model.train()
        train_dataloader.sampler.set_epoch(epoch) # Set epoch for distributed sampler
        total_loss = torch.zeros(3, device=device)
        total_count = torch.tensor(0, device=device)
        
        # Cyclical KL Divergence Annealing
        cycle_length = 500
        iteration = 0
        
        for batch in train_dataloader:
            beta = get_beta_cyclical(iteration, cycle_length, min_beta=0.1, max_beta=1.0)
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
             # AMP Autocast
            with autocast(device_type='cuda', enabled=True):
                recon_batch, mu, log_var = model(batch)
                bce, kl, loss = model.module.loss_function(batch, recon_batch, mu, log_var, beta)
                
            # Scaled backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += torch.tensor((bce.detach(), kl.detach(), loss.detach()), device=device)
            total_count += batch.size(0)
            iteration+=1
            
            del recon_batch, mu, log_var
            torch.cuda.empty_cache()
        # Sum up the losses from all GPUs
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_count, dst=0, op=dist.ReduceOp.SUM)
            
        if local_rank == 0:
            
            avg_loss = total_loss / total_count
            train_history.append({'bce_loss': avg_loss[0].item(), 'kl_loss': avg_loss[1].item(), 'total_loss': avg_loss[2].item()})
            
            if (epoch + 1) % 20 == 0:
                ckpt = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': avg_loss[2],
                }
                path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch+1}.pth')
                torch.save(ckpt, path)
                train_history_dict = compute_dict(train_history)
                validation_history_dict = compute_dict(validation_history)
                plot_history(
                    train_history_dict, validation_history_dict, num_epochs,
                    ckpt_dir, seed)
                print(f"[Rank 0] Saved checkpoint: {path} @ epoch {epoch + 1}")
            
    # # save model
    # model.module.save_model(ckpt_dir)
    dist.barrier()
    dist.destroy_process_group()
            
def eval(model, model_config, val_dataloader, ddp_config):
    torch.cuda.empty_cache()
    ckpt_dir = model_config['ckpt_dir']
    device = ddp_config['device']
    local_rank = ddp_config['local_rank']
    seed = model_config['seed']
    
    # save examples
    example_saved = False
    
    # Load best checkpoint
    if local_rank == 0:
        best_ckpt = torch.load(os.path.join(ckpt_dir, 'best_ckpt.pth'))
        model.module.load_state_dict(best_ckpt['model_state_dict'], strict=False)
        
    # print(torch.cuda.memory_summary(device))
    model.eval()
    set_seed(seed)
    
    # Metrics
    total_psnr = torch.tensor(0.0, device=device)
    total_nrmse = torch.tensor(0.0, device=device)
    total_ssim = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0, device=device)
    
    # FID metrics
    # real_features, fake_features = [], []
    inception = inception_v3(weights='DEFAULT') # Inception model for FID calculation
    feat_dim = inception.fc.in_features
    inception.fc = nn.Identity() # Remove the final classification layer
    inception  =  inception.to(device).eval()
    
    total_real_feat = torch.zeros(feat_dim, device=device)
    total_real_outer = torch.zeros(feat_dim, feat_dim, device=device)
    total_fake_feat = torch.zeros(feat_dim, device=device)
    total_fake_outer = torch.zeros(feat_dim, feat_dim, device=device)
    
    if local_rank == 0:
        batch_iter = tqdm(val_dataloader, desc="Evaluating")
        
    else:
        batch_iter = val_dataloader
    
    val_dataloader.sampler.set_epoch(0) # Set epoch for distributed sampler
    with torch.no_grad():
        for batch in batch_iter:
            batch = batch.to(device, non_blocking=True)
            
            recon_batch, _, _ = model(batch)
            
            ##################
            # Collect metrics#
            ##################
            
            # Collect FID features
            batch_preprocessed = inception_preprocess(batch)
            recon_preprocessed = inception_preprocess(recon_batch)
            real_feat = inception(batch_preprocessed).detach()
            fake_feat = inception(recon_preprocessed).detach()
            
            # real_features.append(real_feat)
            # fake_features.append(fake_feat)
            
            # Collect other metrics
            recon_np = recon_batch.detach().cpu().numpy()
            batch_np = batch.detach().cpu().numpy()
            
            psnr_list, nrmse_list, ssim_list = [], [], []
            for o, r in zip(batch_np, recon_np):
                psnr_list.append(peak_signal_noise_ratio(o, r, data_range=1.0))
                nrmse_list.append(normalized_root_mse(o, r))
                ssim_list.append(structural_similarity(o, r, multichannel=False, data_range=1.0, channel_axis=0)) # (C, H, W)
                
            # Sum up the metrics from all GPUs     
            total_psnr += torch.tensor(psnr_list, device=device).sum()
            total_nrmse += torch.tensor(nrmse_list, device=device).sum()
            total_ssim += torch.tensor(ssim_list, device=device).sum()
            total_count += len(batch)
            
            total_real_feat += real_feat.sum(dim=0)
            total_real_outer += real_feat.t() @ real_feat
            total_fake_feat += fake_feat.sum(dim=0)
            total_fake_outer += fake_feat.t() @ fake_feat
            
            # Save examples
            if local_rank == 0 and not example_saved:
                save_examples(torch.from_numpy(batch_np), 
                              torch.from_numpy(recon_np), ckpt_dir)
                example_saved = True
            
            del recon_batch, batch
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary(device))
            # Collect tensors for FID in rank 0 GPU
            # if local_rank == 0:
            #     real_features.append(torch.from_numpy(batch_np))
            #     fake_features.append(torch.from_numpy(recon_np))
                
            
        # Reduce the metrics from all GPUs to rank 0        
        dist.reduce(total_psnr, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_nrmse, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_ssim, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_count, dst=0, op=dist.ReduceOp.SUM)
        
        dist.reduce(total_real_feat, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_real_outer, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_fake_feat, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_fake_outer, dst=0, op=dist.ReduceOp.SUM)
        
        
    if local_rank == 0:
        N = total_count.item()
        # Calculate average metrics
        avg_psnr = total_psnr.item() / N
        avg_nrmse = total_nrmse.item() / N
        avg_ssim = total_ssim.item() / N
        print(f"Avg PSNR: {avg_psnr:.4f}, Avg NRMSE: {avg_nrmse:.4f}, Avg SSIM: {avg_ssim:.4f}")
        print("Calculating FID")
        # FID 
        # real_all = torch.cat(real_features, dim=0)
        # fake_all = torch.cat(fake_features, dim=0)
        # fid_dict = calculate_metrics(
        #     real_all, fake_all,
        #     cuda=torch.cuda.is_available(), isc=False, fid=True
        # )
        # # print(torch.cuda.memory_summary(device))
        # fid_value = fid_dict['frechet_inception_distance']
        
        
        mu_real = total_real_feat.cpu().numpy() / N
        cov_real = total_real_outer.cpu().numpy() / N - np.outer(mu_real, mu_real)
        mu_fake = total_fake_feat.cpu().numpy() / N
        cov_fake = total_fake_outer.cpu().numpy() / N - np.outer(mu_fake, mu_fake)
        
        fid_value = frechet_distance(mu_real, cov_real, mu_fake, cov_fake)

        # Summary
        metrics = {
            'PSNR': avg_psnr,
            'NRMSE': avg_nrmse,
            'SSIM': avg_ssim,
            'FID': fid_value
        }
    
        print("== Evaluation Results ==")
        for name, value in metrics.items():
            print(f"{name:6s}: {value:.4f}")
    
    # Wait for all processes to finish before exiting
    dist.barrier() 
    # End distributed processing
    dist.destroy_process_group()
    
def get_config(args):
    
    in_channels = 1
    
    model_config = {
        'embed_dim': args['embed_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'weight_decay': args['weight_decay'],
        'lr': args['lr'],
        'img_scale_factor': args['img_scale_factor'],
        'num_epochs': args['num_epochs'],
        'ckpt_dir': args['ckpt_dir'],
        'seed': args['seed'],
        
        # Default values
        'num_heads': 8,
        'num_layers': 6,
        'latent_dim': 128,
        'patch_size': 16,
        'img_size': (480, 640),
        'in_channels': in_channels,
        'model_type': args['model_type']# 'CNN' or 'Transformer'
        
    }
    
    return {
        'task_name': args['task_name'],
        'batch_size': args['batch_size'],
        'img_scale_factor': args['img_scale_factor'],
        'eval': args['eval'],
        'model_config': model_config,
        'num_workers': max(os.cpu_count() - 2 , 1),
        # 'num_workers': 0,
        'in_channels': in_channels,
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE model")
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    
    parser.add_argument('--model_type', action='store', type=str, help='model_type', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    
    parser.add_argument('--img_scale_factor', action='store', type=int, help='img_scale_factor', required=True)
    parser.add_argument('--embed_dim', action='store', type=int, help='embed_dim', required=True)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=True)
    parser.add_argument('--weight_decay', action='store', type=float, help='weight_decay', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    
    
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--eval', action='store_true')
    
    
    main(vars(parser.parse_args()))


            