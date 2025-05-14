import torch
import argparse
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count

from train import get_config

from utils import set_seed, load_data, set_distributed_training, initialize_model_ddp, plot_history, save_examples
from utils import frechet_distance, get_beta_cyclical, inception_preprocess, compute_dict
from models.VAE import TransformerVAE, CNNVAE
# Count Flops


def main(args):
    
    config = get_config(args)
    
    # Get config
    config = get_config(args)
    model_config = config['model_config']
    img_size = model_config['img_size']
    
    image_size = model_config['img_size']
    in_channels = model_config['in_channels']
    img_scale_factor = model_config['img_scale_factor']
    patch_size = model_config['patch_size']
    
    model = TransformerVAE(img_size=image_size, patch_size=patch_size, latent_dim=128, 
                           in_channels=in_channels, img_scale_factor=img_scale_factor)
    
    flops = count_flops(model, in_channels, img_size)
    params = count_params(model)
    
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")

def count_flops(model, in_channels, img_size):
    # Create a dummy input tensor with the same shape as your model's input
    dummy_input = torch.randn(1, in_channels, img_size[0], img_size[1])
    
    # Use FlopCountAnalysis to count the number of FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    
    return flops.total()

def count_params(model):
    # Use parameter_count to count the number of parameters
    params = parameter_count(model) 
    
    return params['']  # Return the total number of parameters
    # Count FLOPs













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