import torch
import os
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as F
from skimage.filters import threshold_otsu
from model import XAIResUNet

logger = logging.getLogger('xairesunet_logger')

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, augment=False):
        self.dataframe = data
        self.transform = transform if transform is not None else self.default_transform
        self.augment = augment

    def __len__(self):
        return len(self.dataframe) * (4 if self.augment else 1)

    def __getitem__(self, index):
        if self.augment:
            original_index = index // 4
            augment_type = index % 4
        else:
            original_index = index
        
        row = self.dataframe.iloc[original_index]
        img_path = row['filename']
        mask_path = row['mask']
        center = row['center']
        patient_id = row['patient_id']
        
        # Extract plane from filename
        plane_img = img_path.split('/')[-1].split('_')[4]
        plane_mask = mask_path.split('/')[-1].split('_')[4]

        if plane_img != plane_mask:
            raise ValueError("Mismatch between image and mask planes")

        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError as e:
            logger.error(f"Error loading image or mask: {e}")
            raise e

        if self.augment and self.augmentation_transforms:
            image, mask = self.augmentation_transforms(image, mask, augment_type)
        else:
            image, mask = self.transform(image, mask)

        thresh = threshold_otsu(np.array(mask))
        binary_mask = np.array(mask) > thresh

        return image, binary_mask, center, patient_id, plane_img
    
    def default_transform(self, image, mask):
        image = F.resize(image, (256, 256))
        mask = F.resize(mask, (256, 256))
        image = F.to_tensor(image)
        mask = F.to_tensor(mask).float()

        # Apply Otsu thresholding to binarize the mask
        mask_np = np.array(mask)
        thresh = threshold_otsu(mask_np)
        binary_mask = mask_np > thresh
        binary_mask = torch.tensor(binary_mask, dtype=torch.float32)

        return image, binary_mask

# Helper function to calculate metrics
def dice_coefficient(pred, true, k=1):
    intersection = np.sum((pred == k) & (true == k)) * 2.0
    pred_sum = np.sum(pred == k)
    true_sum = np.sum(true == k)

    if pred_sum + true_sum == 0:
        return 1.0  # Both are empty, Dice score is perfect

    dice = intersection / (pred_sum + true_sum)
    return dice

def seg_acc(pred_mask, gt_mask):
    eps = 1e-10
    p = pred_mask.astype(int)
    g = gt_mask.astype(int)

    TP = np.sum((p == 1) & (g == 1))
    FP = np.sum((p == 1) & (g == 0))
    TN = np.sum((p == 0) & (g == 0))
    FN = np.sum((p == 0) & (g == 1))

    seg_acc = (TP + TN) / (TP + TN + FP + FN + eps) * 100

    if TP + FP == 0:
        precision = 1.0  # Both predicted and ground truth masks are empty
    else:
        precision = TP / (TP + FP + eps)

    if TP + FN == 0:
        recall = 1.0  # Both predicted and ground truth masks are empty
    else:
        recall = TP / (TP + FN + eps)

    if precision + recall == 0:
        Fscore = 0.0
    else:
        Fscore = 2 * (precision * recall) / (precision + recall + eps)

    if TP + FP + FN == 0:
        Jaccard = 1.0  # Both predicted and ground truth masks are empty
    else:
        Jaccard = TP / (TP + FP + FN + eps)
    
    DSC = dice_coefficient(pred_mask, gt_mask, k=1)

    return seg_acc, precision, recall, Fscore, Jaccard, DSC

def calculate_metrics(preds, labels):
    try:
        logging.info(f"Entering calculate_metrics with preds shape: {preds.shape}, labels shape: {labels.shape}")
        assert preds.dim() == 5 and labels.dim() == 5, "Predictions and labels must be 5D tensors (Batch, Channel, Depth, Height, Width)"
        assert preds.size(0) == labels.size(0), "Batch sizes of predictions and labels must match"
        assert preds.size(2) == labels.size(2) and preds.size(3) == labels.size(3) and preds.size(4) == labels.size(4), "Dimensions of predictions and labels must match"
        
        preds = (preds > 0.5).float()
        labels = labels.float()

        metrics_list = []
        total_false_positives = 0
        total_false_negatives = 0

        for i in range(preds.size(0)):
            pred = preds[i].cpu().numpy().squeeze()
            label = labels[i].cpu().numpy().squeeze()

            if pred.shape != label.shape:
                logging.error(f"Shape mismatch: pred shape {pred.shape}, label shape {label.shape}")
                raise ValueError(f"Shape mismatch: pred shape {pred.shape}, label shape {label.shape}")

            seg_accuracy, precision, recall, f1_score, jaccard, dice_score = seg_acc(pred, label)
            
            metrics_list.append({
                'IoU': jaccard,
                'Dice': dice_score,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Segmentation Accuracy': seg_accuracy
            })

            false_positives = np.sum((pred == 1) & (label == 0))
            false_negatives = np.sum((pred == 0) & (label == 1))
            total_false_positives += false_positives
            total_false_negatives += false_negatives

        metrics_aggregate = {
            'IoU': np.mean([m['IoU'] for m in metrics_list]),
            'Dice': np.mean([m['Dice'] for m in metrics_list]),
            'Precision': np.mean([m['Precision'] for m in metrics_list]),
            'Recall': np.mean([m['Recall'] for m in metrics_list]),
            'F1 Score': np.mean([m['F1 Score'] for m in metrics_list]),
            'Segmentation Accuracy': np.mean([m['Segmentation Accuracy'] for m in metrics_list]),
            'False Positives': total_false_positives,
            'False Negatives': total_false_negatives,
        }

        logger.info(f"Calculated metrics: {metrics_aggregate}")
        return metrics_aggregate

    except AssertionError as e:
        logger.error(f"AssertionError in calculate_metrics: {e}")
        return {
            'IoU': 0,
            'Dice': 0,
            'Precision': 0,
            'Recall': 0,
            'F1 Score': 0,
            'Segmentation Accuracy': 0,
            'False Positives': 0,
            'False Negatives': 0,
        }
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return {
            'IoU': 0,
            'Dice': 0,
            'Precision': 0,
            'Recall': 0,
            'F1 Score': 0,
            'Segmentation Accuracy': 0,
            'False Positives': 0,
            'False Negatives': 0,
        }

def load_latest_checkpoint(model_name, device):
    checkpoint_dir = f'models/{model_name}/model_checkpoints'
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model = XAIResUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

def reconstruct_3D_and_evaluate(patient_id, predictions, ground_truths, planes):
    logger.info(f"Number of prediction slices: {len(predictions)}, Number of ground truth slices: {len(ground_truths)}")
    
    sorted_indices = np.argsort(planes)
    sorted_predictions = [predictions[i] for i in sorted_indices]
    sorted_ground_truths = [ground_truths[i] for i in sorted_indices]
    
    if planes[0] == 'axial':
        stack_dim = 1
    elif planes[0] == 'coronal':
        stack_dim = 2
    elif planes[0] == 'sagittal':
        stack_dim = 0
    else:
        raise ValueError(f"Unknown plane type: {planes[0]}")

    pred_volume = torch.stack(sorted_predictions, dim=stack_dim)
    gt_volume = torch.stack(sorted_ground_truths, dim=stack_dim)
    
    logger.info(f"Shapes after stacking - pred_volume: {pred_volume.shape}, gt_volume: {gt_volume.shape}")

    # [Batch, Channel, Depth, Height, Width]
    if pred_volume.dim() == 4:
        pred_volume = pred_volume.unsqueeze(0)
    if gt_volume.dim() == 4:
        gt_volume = gt_volume.unsqueeze(0)

    logger.info(f"Shapes before metrics calculation - Predictions: {pred_volume.shape}, Ground Truths: {gt_volume.shape}")

    metrics = calculate_metrics(pred_volume, gt_volume)
    
    return metrics

def process_patient(model_name, model, device, center, patient_id, group, criterion, batch_size):
    if center == 'Center_07' and patient_id == 'Patient_08':
        logger.info(f"Excluding patient {patient_id} at center {center}")
        return None
    
    predictions = []
    ground_truths = []
    planes = []

    # Filter the data for the specific patient and center
    patient_data = group[(group['center'] == center) & (group['patient_id'] == patient_id)]
    patient_dataset = CustomDataset(patient_data, transform=None, augment=False)
    patient_loader = DataLoader(patient_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=12)

    for images, masks, centers, patient_ids, planes_batch in patient_loader:
        images, masks = images.to(device), masks.to(device)
        planes.extend(planes_batch)

        with torch.no_grad():
            outputs = model(images)
        preds = torch.sigmoid(outputs).cpu()

        predictions.extend(preds)
        ground_truths.extend(masks.cpu())

    # Reconstruct 3D and evaluate
    patient_metrics = reconstruct_3D_and_evaluate(patient_id, predictions, ground_truths, planes)
    patient_metrics['Center'] = center
    patient_metrics['PatientID'] = patient_id
    patient_metrics['Model'] = model_name

    logger.info(f"Processed patient {patient_id} at center {center} with model {model_name}: {patient_metrics}")

    return patient_metrics

def evaluate_patient_by_patient(model_name, test_file, criterion, device, batch_size=64):
    # Load the test data
    test_data = pd.read_csv(test_file)
    
    grouped = test_data.groupby(['center', 'patient_id'])

    results = []
    model_averages = {}
    center_03_averages = []  
    model = load_latest_checkpoint(model_name, device)

    for (center, patient_id), group in grouped:
        logger.info(f"Processing patient {patient_id} at center {center} with model {model_name}")
        patient_metrics = process_patient(model_name, model, device, center, patient_id, group, criterion, batch_size)
        if patient_metrics is not None:  # Check if patient was processed
            results.append(patient_metrics)

            if center == 'Center_03':
                center_03_averages.append(patient_metrics) 

    # Calculate average metrics for the model
    df_model_results = pd.DataFrame(results)
    model_avg = df_model_results.drop(columns=['Center', 'PatientID', 'Model']).mean().to_dict()
    model_avg['Model'] = model_name
    model_averages[model_name] = model_avg

    # Calculate average metrics for Center_03
    if center_03_averages:
        df_center_03_results = pd.DataFrame(center_03_averages)
        center_03_avg = df_center_03_results.drop(columns=['Center', 'PatientID', 'Model']).mean().to_dict()
        center_03_avg['Model'] = model_name
        center_03_avg['Center'] = 'Center_03'  # Adding this line to keep track of the center
        center_03_averages.append(center_03_avg)

    # Save patient-level results to a single CSV file
    results_dir = f'models/{model_name}/test_results'
    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(results_dir, 'patient_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved patient-level results in {results_csv_path}")

    # Save model average metrics to a separate CSV file
    model_averages_df = pd.DataFrame(model_averages).T.reset_index(drop=True)
    model_averages_csv_path = os.path.join(results_dir, 'model_averages.csv')
    model_averages_df.to_csv(model_averages_csv_path, index=False)
    logger.info(f"Saved model average metrics in {model_averages_csv_path}")

    # Save Center_03 average metrics to a separate CSV file
    if center_03_averages:
        center_03_averages_df = pd.DataFrame(center_03_averages).reset_index(drop=True)
        center_03_averages_csv_path = os.path.join(results_dir, 'center_03_averages.csv')
        center_03_averages_df.to_csv(center_03_averages_csv_path, index=False)
        logger.info(f"Saved Center_03 average metrics in {center_03_averages_csv_path}")



