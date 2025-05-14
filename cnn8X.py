import os, glob, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import normalized_root_mse as nrmse_metric
from datasets import load_dataset
from pytorch_fid import fid_score
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

# 프로젝트 목적에 부합하는 데이터 준비
base = "/Users/huiho/Documents/UMN/Spring2025/CSCI5527_DeepLearning/FinalProject"
all_files = sorted(glob.glob(os.path.join(base, "file-*.parquet")))
print(f"Found {len(all_files)} parquet files in: {base}")

n_train = min(1, len(all_files))  # 현재는 하나의 파일만 사용
train_files = random.sample(all_files, n_train)
test_file = all_files[0]

train_ds = load_dataset("parquet", data_files=train_files, split="train")
test_ds = load_dataset("parquet", data_files=test_file, split="train").select([0])

img_key = "observation.images.top"
train_images = [np.array(ex[img_key], np.float32) / 255.0 for ex in train_ds.select(range(100))]
test_images = [np.array(ex[img_key], np.float32) / 255.0 for ex in test_ds]

class NumpyImageDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.images = image_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, 0

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

train_dataset = NumpyImageDataset(train_images, transform=transform)
test_dataset = NumpyImageDataset(test_images, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} image")

# Autoencoder 모델 정의
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 16, 3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, 3, stride=2, padding=1),
            # nn.ReLU()
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 14x14 → 7x7
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 7x7 → 3x3
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
            # nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 3x3 → 6x6
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),  # 6x6 → 12x12
            # nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),   # 12x12 → 24x24
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# 모델 설정
model = ConvAutoencoder()
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 훈련
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for images, _ in trainloader:
        images = images.to(device)
        output, _ = model(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(trainloader):.4f}")

# 테스트 및 평가 지표
model.eval()
h, w = 60, 80
FID_REAL = "fid_temp/real"
FID_FAKE = "fid_temp/fake"
os.makedirs(FID_REAL, exist_ok=True)
os.makedirs(FID_FAKE, exist_ok=True)

psnr_list, ssim_list, nrmse_list = [], [], []
with torch.no_grad():
    for batch_idx, (images, _) in enumerate(testloader):
        images = images.to(device)
        output, encoded = model(images)

        resized_original = F.interpolate(images, size=(h, w), mode='bilinear', align_corners=False)
        resized_output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

        for i in range(images.size(0)):
            save_image(resized_original[i], f"{FID_REAL}/{batch_idx}_{i}.png")
            save_image(resized_output[i], f"{FID_FAKE}/{batch_idx}_{i}.png")

            orig_np = resized_original[i, 0].cpu().numpy()
            recon_np = resized_output[i, 0].cpu().numpy()
            psnr_list.append(psnr_metric(orig_np, recon_np, data_range=1.0))
            ssim_list.append(ssim_metric(orig_np, recon_np, data_range=1.0))
            nrmse_list.append(nrmse_metric(orig_np, recon_np))

# 시각화
with torch.no_grad():
    sample_images, _ = next(iter(testloader))
    sample_image = sample_images[0:1].to(device)
    reconstructed, encoded = model(sample_image)

latent_np = encoded[0].mean(dim=0).cpu().numpy()
original_np = sample_image.cpu().squeeze().numpy()
reconstructed_np = reconstructed.cpu().squeeze().numpy()

fid_val = fid_score.calculate_fid_given_paths(
    [FID_REAL, FID_FAKE], batch_size=1, device=device, dims=2048, num_workers=0)

print("=== Evaluation ===")
print(f"PSNR: {np.mean(psnr_list):.2f}")
print(f"SSIM: {np.mean(ssim_list):.4f}")
print(f"NRMSE: {np.mean(nrmse_list):.4f}")
print(f"FID: {fid_val:.2f}")

# 모델 복잡도 분석
model.eval()
macs, params = get_model_complexity_info(
    model,
    (1, 480, 640),
    as_strings=True,
    print_per_layer_stat=False
)
print(f"[ptflops] MACs (FLOPs): {macs}")
print(f"[ptflops] Parameters: {params}")

# fig, axs = plt.subplots(1, 3, figsize=(12, 4))

plt.figure(figsize=(4, 3))
plt.title("Original")
plt.imshow(original_np, cmap='gray')
plt.axis('off')

plt.figure(figsize=(4, 3))
plt.title("Latent")
plt.imshow(latent_np, cmap='gray')
plt.axis('off')

plt.figure(figsize=(4, 3))
plt.title("Reconstructed")
plt.imshow(reconstructed_np, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


