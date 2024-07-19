import pandas as pd
from utils.custom_transforms import CustomTransforms
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage.filters import threshold_otsu

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, augment=False):
        if isinstance(data, pd.DataFrame):
            self.dataframe = data
        else:
            self.dataframe = pd.read_csv(data)
        
        self.transform = transform if transform is not None else self.default_transform
        self.augment = augment
        self.augmentation_transforms = CustomTransforms(alpha=100.0) if augment else None

    def __len__(self):
        return len(self.dataframe) * (4 if self.augment else 1)

    def __getitem__(self, index):
        if self.augment:
            original_index = index // 4
            augment_type = index % 4
        else:
            original_index = index
            augment_type = None
        
        img_path = self.dataframe.iloc[original_index]['filename']
        mask_path = self.dataframe.iloc[original_index]['mask']
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError as e:
            logging.error(f"Error loading image or mask: {e}")
            raise e

        if self.augment and self.augmentation_transforms:
            image, mask = self.augmentation_transforms(image, mask, augment_type)
        else:
            image, mask = self.transform(image, mask)

        thresh = threshold_otsu(np.array(mask))
        binary_mask = np.array(mask) > thresh

        return image, binary_mask, index
    
    def default_transform(self, image, mask):
        image = F.resize(image, (256, 256))
        mask = F.resize(mask, (256, 256))
        image = F.to_tensor(image)
        mask = F.to_tensor(mask).float()

        # Apply Otsu thresholding to binarize the mask
        mask_np = np.array(mask)
        thresh = threshold_otsu(mask_np)
        binary_mask = mask_np > thresh
        binary_mask = torch.tensor(binary_mask, dtype=torch.float32)  # Convert back to tensor

        return image, binary_mask