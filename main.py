import sys
import os
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader
import wandb
import argparse
from scripts.train import train_model
from scripts.test import evaluate_patient_by_patient, load_latest_checkpoint
from model import XAIResUNet
from utils.custom_transforms import CustomTransforms
from utils.custom_dataset import CustomDataset
from utils.logging_setup import setup_logging
from monai.losses import DiceFocalLoss
from utils.cam import generate_and_save_cam_images

def initialize_wandb(model_name):
    wandb.init(project=f"imvip_{model_name.lower()}", mode="online")

def load_model(model_name, device):
    if model_name == "XAIResUnet_Imagenet":
        model = XAIResUNet(pretrained=True, num_classes=1, weights_path=None, dropout_rate=0.5)
    elif model_name == "XAIResUnet_Radimagenet":
        model = XAIResUNet(pretrained=False, weights_path='ResNet50.pt', num_classes=1, dropout_rate=0.5)
    elif model_name == "XAIResUnet_Vanilla":
        model = XAIResUNet(pretrained=False, num_classes=1, weights_path=None, dropout_rate=0.5)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)

def main(model_name, epochs):
    logger = setup_logging(model_name)
    # initialize_wandb(model_name)

    data_dir = '/home/vganepola/IMVIP/v2/XAI-ResUnet/data/preprocessed_data'
    train_file = os.path.join(data_dir, 'train.csv')
    validation_file = os.path.join(data_dir, 'validation.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device set to: {device}")

    model = load_model(model_name, device)

    # Data loading
    train_dataset = CustomDataset(train_file, transform=CustomTransforms(alpha=100.0), augment=True)
    val_dataset = CustomDataset(validation_file, transform=None, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

     # Training and validation
    best_model_path = train_model(model, train_loader, val_loader, device, model_name, num_epochs=epochs)

    # Load the best model for testing
    model = load_latest_checkpoint(model_name, device)

    # Testing
    evaluate_patient_by_patient(model_name, test_file, criterion=DiceFocalLoss(sigmoid=True), device=device, batch_size=16)

    # CAM generation
    specific_indices = [173 * 32 + 29, 173 * 32 + 30]
    output_dir = f'models/{model_name}/test_results/cam'
    os.makedirs(output_dir, exist_ok=True)

    # can pass multiple models if you want
    models = {
        model_name: model
    }

    generate_and_save_cam_images(models, test_file, specific_indices, device, output_dir)

    logger.handlers[0].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('--model', type=str, required=True, help='Model name (XAIResUnet_Imagenet/XAIResUnet_Radimagenet/XAIResUnet_Vanilla)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for training')

    args = parser.parse_args()

    main(args.model, args.epochs)
