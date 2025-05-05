from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms.v2 as v2
import numpy as np


class RobotDataset(Dataset):
    def __init__(self, dataset, split='train', norm_stats=None, image_size = (480,640)):
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.image_size = image_size
        # self.image_saling_factor = image_scaling_factor
        train_indices, val_indices = self.split_indices(ratio=0.8)
        
        # self.transforms_href = self.get_transform()
        # self.transforms_lref = self.get_transform(scaling=True)
        self.transforms = self.get_transform()
        if split == 'train':
            self.dataset = self.dataset.select(train_indices)
        elif split == 'val':
            self.dataset = self.dataset.select(val_indices)
        else:
            return None
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample.get('observation.images.top')
        # image = image.convert('RGB') 
        # image = (image - self.norm_stats['mean']) / self.norm_stats['std'] # Normalize the image
        image = self.transforms(image) # Apply the transform
        
        return image
        
        image_orig = self.transforms_href(image) # Apply the transform
        image_scaled = self.transforms_lref(image)
        
        return {
            'orig': image_orig,
            'scaled': image_scaled,
        }
        
    def split_indices(self, ratio=0.8):
        """
        Split the dataset into training and validation sets.
        Returns:
            train_indices: The indices for the training set.
            val_indices: The indices for the validation set.
        """
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        split = int(np.floor(ratio * num_samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        
        return train_indices, val_indices
    
    def get_transform(self):
        """
        Get the transform for the dataset.
        Returns:
            transform: The transform to apply to the dataset.
        """
        
        return v2.Compose([
            # v2.ToImage(), # Convert from PIL to tensor
            v2.ToDtype(torch.float32, scale=True), # float32 in [0, 1]
            # v2.Resize(size=new_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Normalize(mean=self.norm_stats['mean'], std=self.norm_stats['std']),
        ])
        
        
        