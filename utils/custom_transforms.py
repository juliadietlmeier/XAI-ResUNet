import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import numpy as np
from skimage.filters import threshold_otsu

class CustomTransforms:
    def __init__(self, alpha=50.0):
        self.alpha = alpha
        self.resize = v2.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)

    def elastic_transform(self, image, mask):
        elastic_transform = v2.ElasticTransform(alpha=self.alpha)
        return elastic_transform(image, mask)
    
    def vertical_flip(self, image, mask):
        return F.vflip(image), F.vflip(mask)
    
    def horizontal_flip(self, image, mask):
        return F.hflip(image), F.hflip(mask)
    
    def apply_color_jitter(self, image):
        return self.color_jitter(image)

    def __call__(self, image, mask, transform_type):
        if transform_type == 0:
            image, mask = self.elastic_transform(image, mask)
        elif transform_type == 1:
            image, mask = self.vertical_flip(image, mask)
        elif transform_type == 2:
            image, mask = self.horizontal_flip(image, mask)
        elif transform_type == 3:
            image = self.apply_color_jitter(image)
        
        image = self.resize(image)
        mask = self.resize(mask)
        
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        mask_np = np.array(mask)
        thresh = threshold_otsu(mask_np)
        binary_mask = mask_np > thresh
        binary_mask = torch.tensor(binary_mask, dtype=torch.float32)  # Convert back to tensor

        return image, binary_mask
