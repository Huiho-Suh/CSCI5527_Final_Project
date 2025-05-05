import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset, Image
from data_loader import RobotDataset
from torch.utils.data import DataLoader, DistributedSampler
from models.VAE import TransformerVAE, CNNVAE
import os
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt

# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Mixed Precision Training
from torch.amp import GradScaler
import torch.optim as optim


def get_norm_stats(dataset, in_channels=3):
    """
    Get the mean and std of the dataset.
    Args:
        dataset: The dataset to get the mean and std from. (in torch format)
    Returns:
        mean: The mean of the dataset.
        std: The std of the dataset.
    """
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_pixels = 0
    sum_ = torch.zeros(in_channels, device=device)
    sum_sq = torch.zeros(in_channels, device=device)
    
    with torch.no_grad():
        for batch in loader:
            images = batch['observation.images.top'].to(device)
                
            images = v2.ToDtype(torch.float32, scale=True)(images) # Convert to float32 in [0,1]
            
            B, C, H, W = images.shape
            n_pixels += B * H * W
            sum_ += images.sum(dim=(0, 2, 3))
            sum_sq += (images ** 2).sum(dim=(0, 2, 3))
            
    mean = sum_ / n_pixels
    std = torch.sqrt(sum_sq / n_pixels - mean ** 2)
    mean = mean.cpu()
    std = std.cpu()
    
    norm_stats = {
        'mean': mean,
        'std': std
    }
    
    return norm_stats

def get_transforms(mean, std, scale_factor, image_size):
    """
    Get the transforms for the dataset.
    Args:
        mean: The mean of the dataset.
        std: The std of the dataset.
        scale_factor: The scale factor for resizing.
        image_size: The size of the images.
    Returns:
        transform: The transform to apply to the dataset.
    """
    
    transform = v2.Compose([
        v2.Resize((int(image_size[0] * scale_factor), int(image_size[1] * scale_factor))),
        v2.Normalize(mean, std),
        v2.ToTensor(),
    ])
    
    return transform

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_data(config):
    
    task_name = config['task_name']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    in_channels = config['in_channels']
    # img_scale_factor = config['img_scale_factor']
    
    
    # Load dataset
    robot_dataset = load_dataset(f"lerobot/{task_name}", split="train")
    # robot_dataset = robot_dataset.cast_column("observation.images.top", Image(decode=True, mode="RGB")) # Forcing the image to be a PIL image
    robot_dataset = robot_dataset.with_format("torch", columns=["observation.images.top"]) # Change to torch format
    robot_dataset = robot_dataset.cast_column("observation.images.top", Image(decode=True, mode="L")) # Forcing the image to be a PIL image
    norm_stats = get_norm_stats(robot_dataset, in_channels=in_channels)
    
    train_dataset = RobotDataset(robot_dataset, split='train', norm_stats=norm_stats)
    train_sampler = DistributedSampler(train_dataset) # Distributed Sampler
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                sampler=train_sampler, pin_memory=True)
    
    val_dataset = RobotDataset(robot_dataset, split='val', norm_stats=norm_stats)
    val_sampler = DistributedSampler(val_dataset) # Distributed Sampler
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                sampler=val_sampler, pin_memory=True)
    
    return train_dataloader, val_dataloader, norm_stats
    
    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'train_sampler': train_sampler,
        'val_sampler': val_sampler,
        'norm_stats': norm_stats
    }
    
def make_model(model_config):
    """
    Create a model based on the given configuration.
    Args:
        model_config: The configuration for the model.
    Returns:
        model: The created model.
    """
    
    # Create the model
    model = TransformerVAE(**model_config)
    
    return model

def set_distributed_training():
    """
    Set up distributed training.
    """
    
    # Distributed training setup
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    
    return {'local_rank': local_rank, 'device': device}

def initialize_model_ddp(model_config, ddp_config):
    
    lr = model_config['lr']
    weight_decay = model_config['weight_decay']
    device = ddp_config['device']
    image_size = model_config['img_size']
    in_channels = model_config['in_channels']
    img_scale_factor = model_config['img_scale_factor']
    patch_size = model_config['patch_size']
    # Only set this if you are using a GPU with limited memory
    # torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    
    if model_config['model_type'] == 'CNN':
        model = CNNVAE(in_channels=in_channels, hidden_channels=64, latent_channels=128, scale=img_scale_factor).to(device)
    elif model_config['model_type'] == 'Transformer':
        model = TransformerVAE(img_size=image_size, patch_size=patch_size, latent_dim=128, in_channels=in_channels, img_scale_factor=img_scale_factor).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = DDP(model, device_ids=[ddp_config["local_rank"]], output_device=ddp_config["local_rank"]) # DDP Wrapping
    model = torch.compile(model) # Torch Compile Wrapping
    scaler = GradScaler() # Automatic Mixed Precision Scaler
    
    return model, optimizer, scaler

def plot_history(train_history_dict, val_history_dict, num_epochs, ckpt_dir, seed):
    
    for k in train_history_dict.keys():
        train_values = train_history_dict[k]
        val_values = val_history_dict[k]
        # save training curves
        plot_path = os.path.join(ckpt_dir, f'{k}_seed_{seed}.png')
        plt.figure()
        plt.plot(np.linspace(0, num_epochs-1, len(train_values)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(val_values)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(f"k")
        plt.savefig(plot_path)
        print(f'Saved plots to {ckpt_dir}')
    
def save_examples(orig_batch, recon_batch, save_dir):
    
    random_idx = np.random.randint(0, orig_batch.shape[0])
    os.makedirs(save_dir, exist_ok=True)
    orig_tensor = orig_batch[random_idx]
    recon_tensor = recon_batch[random_idx]
    
    # Convert to [0,1] range
    recon_tensor = torch.sigmoid(recon_tensor)
    # (C, H, W) -> (H, W, C)
    orig = (orig_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    recon = (recon_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(recon, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'orig_recon.png'))
    plt.close(fig)  

def frechet_distance(mu1: np.ndarray,
                     sigma1: np.ndarray,
                     mu2: np.ndarray,
                     sigma2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """
    Fréchet Inception Distance 계산.
    mu*: 특징(feature) 평균, shape=(D,)
    sigma*: 특징 공분산(covariance), shape=(D,D)
    """
    diff = mu1 - mu2
    # sqrt of product of covariances
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # 수치 안정화
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real(covmean)  # 복소수 오차 제거
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))

def get_beta_cyclical(iteration, cycle_length, min_beta, max_beta):
    """
    Get beta value for cyclical learning rate.
    Args:
        iteration: Current iteration.
        cycle_length: Length of the cycle.
        min_beta: Minimum beta value.
        max_beta: Maximum beta value.
    Returns:
        beta: Beta value for the current iteration.
    """
    
    cycle = np.floor(1 + iteration / (2 * cycle_length))
    x = np.abs(iteration / cycle_length - 2 * cycle + 1)
    beta = min_beta + (max_beta - min_beta) * np.maximum(0, (1 - x))
    
    return beta

def inception_preprocess(image_batch):
    """
    Preprocess images for Inception model.
    Args:
        images: Images to preprocess. (Batch, C, H, W)
    Returns:
        preprocessed_images: Preprocessed images. (Batch, C, H, W)
    """
    
    # Resize and normalize the images
    
    preprocess = v2.Compose([v2.Resize((299,299)),
                             v2.Grayscale(num_output_channels=3),
                             # Use ImageNet mean and std so the activations 
                             # land in the same range as Inception-v3 was trained on
                             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ]) 
    
    
    return preprocess(image_batch)

def compute_dict(history):
    """
    Compute the mean and std of the history dictionary.
    Args:
        history: The history dictionary. (list of dicts)
    Returns:
        history_dict: dictonary of list
    """
    
    history_dict = {k: None for k in history[0].keys()}
    for k in history[0].keys():
        history_dict[k] = [history[i][k] for i in range(len(history))]
        
    return history_dict