import os
import logging
import torch
from monai.losses import DiceFocalLoss
from utils.metrics import calculate_metrics
from utils.visualization import plot_and_save_metrics, log_metrics_to_csv

logger = logging.getLogger('xairesunet_logger')

def train_model(model, train_loader, val_loader, device, model_name, num_epochs=100):
    logger.info("Starting training...")
    criterion = DiceFocalLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    checkpoint_dir = f'models/{model_name}/model_checkpoints'
    plots_dir = f'models/{model_name}/plots'
    logs_dir = f'models/{model_name}/metrics/metrics_log.csv'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = None
    epoch_stats = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}...")
        model.train()
        train_loss = 0
        train_metrics_with_lesions = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}
        train_metrics_without_lesions = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}
        num_batches_with_lesions_train = 0
        num_batches_without_lesions_train = 0
        num_batches = len(train_loader)

        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(preds, masks)

            for i in range(images.size(0)):
                img, mask = images[i], masks[i]
                if mask.sum() > 0:  # Slices with lesions
                    for key in train_metrics_with_lesions:
                        train_metrics_with_lesions[key] += batch_metrics[key]
                    num_batches_with_lesions_train += 1
                else:  # Slices without lesions
                    for key in train_metrics_without_lesions:
                        train_metrics_without_lesions[key] += batch_metrics[key]
                    num_batches_without_lesions_train += 1

        if num_batches_with_lesions_train > 0:
            for key in train_metrics_with_lesions:
                train_metrics_with_lesions[key] /= num_batches_with_lesions_train
        if num_batches_without_lesions_train > 0:
            for key in train_metrics_without_lesions:
                train_metrics_without_lesions[key] /= num_batches_without_lesions_train

        train_loss /= num_batches

        logger.info(f"Training - Epoch: {epoch+1}, Loss: {train_loss:.4f}")
        # wandb.log({"train/loss": train_loss}, step=epoch)
        # wandb.log({f"train_with_lesions/{key}": value for key, value in train_metrics_with_lesions.items()}, step=epoch)
        # wandb.log({f"train_without_lesions/{key}": value for key, value in train_metrics_without_lesions.items()}, step=epoch)

        val_loss = 0
        val_metrics_with_lesions = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}
        val_metrics_without_lesions = {'IoU': 0, 'Dice': 0, 'Precision': 0, 'Recall': 0, 'F1 Score': 0}
        num_batches_with_lesions_val = 0
        num_batches_without_lesions_val = 0

        if val_loader is not None:
            model.eval()
            num_val_batches = len(val_loader)

            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    preds = torch.sigmoid(outputs)
                    batch_metrics = calculate_metrics(preds, masks)

                    for i in range(images.size(0)):
                        img, mask = images[i], masks[i]
                        if mask.sum() > 0:  # Slices with lesions
                            for key in val_metrics_with_lesions:
                                val_metrics_with_lesions[key] += batch_metrics[key]
                            num_batches_with_lesions_val += 1
                        else:  # Slices without lesions
                            for key in val_metrics_without_lesions:
                                val_metrics_without_lesions[key] += batch_metrics[key]
                            num_batches_without_lesions_val += 1

            if num_batches_with_lesions_val > 0:
                for key in val_metrics_with_lesions:
                    val_metrics_with_lesions[key] /= num_batches_with_lesions_val
            if num_batches_without_lesions_val > 0:
                for key in val_metrics_without_lesions:
                    val_metrics_without_lesions[key] /= num_batches_without_lesions_val

            val_loss /= num_val_batches

            scheduler.step(val_loss)

            logger.info(f"Validation - Epoch: {epoch+1}, Loss: {val_loss:.4f}")
            # wandb.log({"val/loss": val_loss}, step=epoch)
            # wandb.log({f"val_with_lesions/{key}": value for key, value in val_metrics_with_lesions.items()}, step=epoch)
            # wandb.log({f"val_without_lesions/{key}": value for key, value in val_metrics_without_lesions.items()}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Best model saved at {best_model_path}")
                # wandb.save(best_model_path)

        epoch_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loader is not None else None,
            'train_metrics': {
                'IoU': train_metrics_with_lesions if num_batches_with_lesions_train > 0 else train_metrics_without_lesions,
                'Dice': train_metrics_with_lesions if num_batches_with_lesions_train > 0 else train_metrics_without_lesions,
                'Precision': train_metrics_with_lesions if num_batches_with_lesions_train > 0 else train_metrics_without_lesions,
                'Recall': train_metrics_with_lesions if num_batches_with_lesions_train > 0 else train_metrics_without_lesions,
                'F1 Score': train_metrics_with_lesions if num_batches_with_lesions_train > 0 else train_metrics_without_lesions,
            },
            'val_metrics': {
                'IoU': val_metrics_with_lesions if num_batches_with_lesions_val > 0 else val_metrics_without_lesions,
                'Dice': val_metrics_with_lesions if num_batches_with_lesions_val > 0 else val_metrics_without_lesions,
                'Precision': val_metrics_with_lesions if num_batches_with_lesions_val > 0 else val_metrics_without_lesions,
                'Recall': val_metrics_with_lesions if num_batches_with_lesions_val > 0 else val_metrics_without_lesions,
                'F1 Score': val_metrics_with_lesions if num_batches_with_lesions_val > 0 else val_metrics_without_lesions,
            } if val_loader is not None else None,
        })

        torch.cuda.empty_cache()

    logger.info("Training completed.")
    
    plot_and_save_metrics(epoch_stats, plots_dir)
    # log_metrics_to_csv(epoch_stats, logs_dir)

    return best_model_path

