from datasets import load_dataset
from PIL import Image
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# dataset = load_dataset("lerobot/aloha_sim_transfer_cube_scripted_image", split="train")
dataset = load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
# dataset = load_dataset("lerobot/aloha_sim_insertion_scripted_image", split="train")

images = dataset['observation.images.top']

for image in tqdm(images):
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        print(f"Unexpected image type: {type(image)}")
        continue

