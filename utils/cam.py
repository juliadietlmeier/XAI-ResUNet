import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage.filters import threshold_otsu
from model import XAIResUNet

# Set up logging
logger = logging.getLogger('xairesunet_logger')

def convert_mask_to_binary_mask(cam_image, threshold=0.5):
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())
    binary_mask = (cam_image > threshold).astype(float)
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[..., 0]
    return binary_mask

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, specific_indices=None):
        logger.info("Initializing CustomDataset")
        self.dataframe = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        if specific_indices:
            logger.info(f"Filtering dataset with specific indices: {specific_indices}")
            self.dataframe = self.dataframe.iloc[specific_indices].reset_index(drop=True)
        self.transform = transform if transform else self.default_transform
        logger.info(f"Dataset size: {len(self.dataframe)}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filename']
        mask_path = self.dataframe.iloc[idx]['mask']
        logger.info(f"Loading image: {img_path} and mask: {mask_path}")
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image, mask = self.transform(image, mask)

        thresh = threshold_otsu(np.array(mask))
        binary_mask = (np.array(mask) > thresh).astype(float)
        binary_mask = torch.tensor(binary_mask, dtype=torch.float32)

        return image, binary_mask, img_path

    def default_transform(self, image, mask):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return transform(image), transform(mask)

class SemanticSegmentationTarget:
    def __init__(self, mask, device):
        self.mask = torch.tensor(mask, dtype=torch.float32).to(device)

    def __call__(self, model_output):
        return (model_output * self.mask).sum()

def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def generate_cam(model, image, mask, device, target_layer_name, cam_method=EigenGradCAM):
    logger.info(f"Generating CAM for target layer: {target_layer_name}")
    target_layer_instance = eval(f"model.{target_layer_name}")
    cam = cam_method(model=model, target_layers=[target_layer_instance])
    target = SemanticSegmentationTarget(mask, device)
    grayscale_cam = cam(input_tensor=image, targets=[target])[0]
    return grayscale_cam

def save_combined_image(models, test_loader, target_layers, device, output_dir):
    logger.info("Starting to save combined CAM images")
    num_images = len(test_loader.dataset)
    num_rows = num_images  # Number of rows determined by the number of specific indices
    fig, axes = plt.subplots(num_rows, len(target_layers) + 3, figsize=(30, num_rows * 5))
    
    row = 0
    for model_name, model in models.items():
        logger.info(f"Processing model: {model_name}")
        for i in range(num_images):
            image, mask, img_path = test_loader.dataset[i]
            image = image.unsqueeze(0).to(device).requires_grad_(True)
            mask = mask.unsqueeze(0).to(device)

            logger.info(f"Running inference for image: {img_path}")
            outputs = model(image)
            pred_mask = torch.sigmoid(outputs).squeeze().cpu().detach().numpy()
            pred_mask_binary = pred_mask > 0.5
            dice_score_pred = dice_coefficient(torch.tensor(pred_mask_binary), mask.cpu().detach())

            for col, layer_name in enumerate(target_layers):
                grayscale_cam = generate_cam(model, image, mask, device, layer_name)

                rgb_img = image[0].detach().cpu().numpy()
                rgb_img = np.moveaxis(rgb_img, 0, -1)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                
                if rgb_img.shape[2] == 1:
                    rgb_img = np.repeat(rgb_img, 3, axis=2)
                
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                if mask.ndim == 4:
                    mask = mask.squeeze()
                if mask.shape != (256, 256):
                    mask = mask.cpu().numpy().squeeze()
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy().squeeze()

                gt_contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cam_image_with_contours = cam_image.copy()
                cv2.drawContours(cam_image_with_contours, gt_contours, -1, (0, 0, 0), 2)

                axes[row, col + 3].imshow(cam_image_with_contours)
                axes[row, col + 3].axis('off')

            axes[row, 0].imshow(image[0].cpu().detach().numpy().transpose(1, 2, 0))
            axes[row, 0].axis('off')

            axes[row, 1].imshow(mask, cmap='gray')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_mask_binary, cmap='gray')
            axes[row, 2].axis('off')
            axes[row, 2].text(10, 10, f'Dice: {dice_score_pred:.4f}', color='yellow', fontsize=20, ha='left', va='top')

            row += 1

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.01, hspace=0.0001)
    output_path = os.path.join(output_dir, 'cam_image.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    logger.info(f"Saved combined CAM image to {output_path}")
    plt.close(fig)

def generate_and_save_cam_images(models, test_csv, specific_indices, device, output_dir):
    logger.info("Generating and saving CAM images")
    target_layers = [
        'enc1',
        'enc2',
        'enc3',
        'enc4',
        'enc5',
    ]
    
    test_dataset = CustomDataset(test_csv, specific_indices=specific_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    save_combined_image(models, test_loader, target_layers, device, output_dir)
    logger.info("CAM images generation completed.")

