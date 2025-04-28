import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import math

torch.set_float32_matmul_precision('high')


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(480, 640), patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1,2) # (B, num_patches, embed_dim)
        x += self.pos_embed
        return x
    
class TransformerVAE(nn.Module):
    def __init__(self, img_size=(480, 640), img_scale_factor=0.5, patch_size=16, in_channels=3, latent_dim=128, num_layers=6, num_heads=8, 
                 embed_dim=768, dim_feedforward=2048):
        super().__init__()
        
        # Parameters
        self.img_size =img_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.img_scale_factor = img_scale_factor
        self.in_channels = in_channels
        
        # The model accepts scaled images and outputs original size images
        self.new_img_size = self.scaled_image_size(img_size, img_scale_factor, patch_size)
        
        self.num_enc_patches = (self.new_img_size[0] // patch_size) * (self.new_img_size[1] // patch_size)
        self.num_dec_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)
        
        self.patch_embedding = PatchEmbedding(self.new_img_size, patch_size, in_channels, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(embed_dim * self.num_enc_patches, latent_dim)
        self.fc_log_var = nn.Linear(embed_dim * self.num_enc_patches, latent_dim)
        
        # Transformer Decoder
        self.decoder_input = nn.Linear(latent_dim, embed_dim * self.num_dec_patches)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_output = nn.Linear(embed_dim, in_channels * patch_size * patch_size) # Recover original image size
        
        # Loss function
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')
        
    def encode(self, x):
        x = self.patch_embedding(x) # (B, num_patches, embed_dim)
        x = self.encoder(x) # (B, num_patches, embed_dim)
        x = x.flatten(1) # (B, num_patches * embed_dim)
        
        mu = self.fc_mu(x) # (B, latent_dim)
        log_var = self.fc_log_var(x) # (B, latent_dim)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode (self, z):
        x = self.decoder_input(z) # (B, num_patches * embed_dim)
        x = x.view(-1, self.num_dec_patches, self.embed_dim) # (B, num_patches, embed_dim)
        x = self.decoder(tgt=x, memory=x) # (B, num_patches, embed_dim)
        x = self.decoder_output(x) # (B, num_patches, patch_size * patch_size * 3)
        x = x.view(-1, self.in_channels, self.img_size[0], self.img_size[1]) # (B, C, H, W)
        
        # return torch.sigmoid(x) # values between 0 and 1
        return x
    
    def forward(self, x):
        # Scale the image
        x = transforms.Resize(self.new_img_size)(x)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        
        return x_recon, mu, log_var
    
    def loss_function(self, x, x_recon, mu, log_var, beta=1.0):
        B = x.size(0)
        # 1) BCE
        logits = x_recon.view(B, -1)
        target = x.view(B, -1).float()
        bce_loss = self.BCE(logits, target)
        # 2) KL
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 3) 배치당 평균 혹은 sum
        return (bce_loss + beta * kl_loss) / B
    
    def scaled_image_size(self, img_size, img_scale_factor, patch_size):
        new_size = (int(img_size[0] * img_scale_factor),
                    int(img_size[1] * img_scale_factor))
        # Ensure the new size is divisible by patch size
        new_size = (new_size[0] // patch_size * patch_size, new_size[1] // patch_size * patch_size)
        
        return new_size
    
    def generate(self, z):
        with torch.no_grad():
            x_recon = self.decode(z)
        return x_recon
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        samples = self.generate(z)
        return samples
    
    def visualize_samples(self, samples, save_path):
        
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
        save_image(grid, save_path)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.show()
        
    def visualize_reconstructions(self, original, reconstructed, save_path):
        grid = torchvision.utils.make_grid(torch.cat([original, reconstructed], dim=0), nrow=8, normalize=True)
        save_image(grid, save_path)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.show()
        
    def calculate_mse(self, original, reconstructed):
        original = original.view(original.size(0), -1)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)
        mse = mean_squared_error(original.cpu().numpy(), reconstructed.cpu().numpy())
        return mse
    
    def plot_mse_distribution(self, mse_values, save_path):
        plt.hist(mse_values, bins=50, density=True)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Density')
        plt.title('MSE Distribution')
        plt.savefig(save_path)
        plt.show()
     