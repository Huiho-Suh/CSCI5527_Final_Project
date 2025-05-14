import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
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
    def __init__(self, img_size=(480, 640), img_scale_factor=2, patch_size=16, in_channels=3, latent_dim=128, num_layers=6, num_heads=8, embed_dim=768, dim_feedforward=2048):
        super().__init__()
        
        # Parameters
        self.img_size =img_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.img_scale_factor = 1/img_scale_factor
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
        x = transforms.Resize(self.new_img_size)(x) # Resize to new image size
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
        bce = self.BCE(logits, target)
        # 2) KL
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # 3) average sum
        loss = bce + beta * kl
        
        return bce, kl, loss

    def scaled_image_size(self, img_size, img_scale_factor, patch_size):
        # Apply scaling factor
        new_H = img_size[0] // img_scale_factor
        new_W = img_size[1] // img_scale_factor
        
        # Ensure the new size is divisible by patch size
        new_H = new_H // patch_size * patch_size
        new_W = new_W // patch_size * patch_size
        
        new_size = (int(new_H), int(new_W))
        
        return new_size
    
import math
import torch.nn.functional as F

class CNNVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        latent_channels=128,
        scale=2,
        num_feat_blocks=3  # Number of feature extraction blocks for depth control
    ):
        super().__init__()
        # Ensure scale is a power of two (2, 4, 8, ...)
        assert scale & (scale - 1) == 0 and scale > 0, "scale must be a power of two."
        n_scales = int(math.log2(scale))

        # ---- Encoder Feature Extraction: stride=1 conv blocks (maintain spatial resolution) ----
        enc_feat = []
        for i in range(num_feat_blocks):
            enc_feat.append(nn.Conv2d(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size=3, stride=1, padding=1
            ))
            enc_feat.append(nn.ReLU(inplace=True))
        self.enc_feat = nn.Sequential(*enc_feat)

        # ---- Encoder Downsampling: stride=2 conv blocks (reduce spatial resolution) × n_scales ----
        enc_down = []
        for _ in range(n_scales):
            enc_down.append(nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ))
            enc_down.append(nn.ReLU(inplace=True))
        self.enc_down = nn.Sequential(*enc_down)

        # ---- Latent Projections: project to mean and log-variance ----
        self.conv_mu     = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)

        # ---- Decoder Preparation: map latent vector back to feature space ----
        self.dec_prep = nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1)

        # ---- Decoder Upsampling: PixelShuffle blocks × n_scales (increase spatial resolution) ----
        dec_up = []
        for _ in range(n_scales):
            dec_up.append(nn.Conv2d(
                hidden_channels, hidden_channels * 4,
                kernel_size=3, padding=1
            ))
            dec_up.append(nn.PixelShuffle(2))
            dec_up.append(nn.ReLU(inplace=True))
        self.dec_up = nn.Sequential(*dec_up)

        # ---- Decoder Feature Refinement: stride=1 conv blocks (maintain spatial resolution) ----
        dec_feat = []
        for _ in range(num_feat_blocks):
            dec_feat.append(nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=3, stride=1, padding=1
            ))
            dec_feat.append(nn.ReLU(inplace=True))
        self.dec_feat = nn.Sequential(*dec_feat)

        # ---- Final Output Convolution: generate high-resolution image ----
        self.final = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def reparameterize(self, mu, logvar):
        # Apply the reparameterization trick to sample from N(mu, sigma^2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_lr):
        # 1) Encoder: extract features and downsample
        h = self.enc_feat(x_lr)    # [B, hidden_channels, H, W]
        h = self.enc_down(h)       # [B, hidden_channels, H/scale, W/scale]

        # 2) Compute latent mean and log-variance, then sample
        mu     = self.conv_mu(h)   # [B, latent_channels, H/scale, W/scale]
        logvar = self.conv_logvar(h)
        z      = self.reparameterize(mu, logvar)

        # 3) Decoder: prepare features, upsample, then refine
        d = self.dec_prep(z)       # [B, hidden_channels, H/scale, W/scale]
        d = self.dec_up(d)         # [B, hidden_channels, H, W]
        d = self.dec_feat(d)       # [B, hidden_channels, H, W]
        x_hr = self.final(d)       # [B, in_channels, H*scale, W*scale]

        return x_hr, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar, beta=1.0):
        # 1) Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        # 2) KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 3) Total loss
        total_loss = recon_loss + beta * kl_loss
        return recon_loss, kl_loss, total_loss
