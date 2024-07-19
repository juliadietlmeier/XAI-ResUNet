import matplotlib.pyplot as plt
import os
import pandas as pd
import logging

logger = logging.getLogger('xairesunet_logger')

def plot_and_save_metrics(epoch_stats, save_dir='model_imagenet/plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score']

    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(
            [stat['epoch'] for stat in epoch_stats],
            [stat['train_metrics'][metric][metric] for stat in epoch_stats],
            label='Train',
            color='blue',
            linestyle='-'
        )
        ax.plot(
            [stat['epoch'] for stat in epoch_stats if stat['val_metrics'] is not None],
            [stat['val_metrics'][metric][metric] for stat in epoch_stats if stat['val_metrics'] is not None],
            label='Validation',
            color='green',
            linestyle='--'
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend(loc='best')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_stats.png'), dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        [stat['epoch'] for stat in epoch_stats],
        [stat['train_loss'] for stat in epoch_stats],
        label='Train Loss',
        color='blue',
        linestyle='-'
    )
    ax.plot(
        [stat['epoch'] for stat in epoch_stats if stat['val_loss'] is not None],
        [stat['val_loss'] for stat in epoch_stats if stat['val_loss'] is not None],
        label='Validation Loss',
        color='green',
        linestyle='--'
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_stats.png'), dpi=300)
    plt.close(fig)

def log_metrics_to_csv(epoch_stats, csv_path='model_imagenet/metrics_log.csv'):
    csv_dir = os.path.dirname(csv_path)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    data = []
    for stat in epoch_stats:
        for key in ['train_metrics', 'val_metrics']:
            entry = {
                'Epoch': stat['epoch'],
                'Phase': 'Train' if key == 'train_metrics' else 'Validation',
                'Loss': stat['train_loss'] if key == 'train_metrics' else stat['val_loss']
            }
            if stat[key] is not None:
                entry.update({
                    'IoU': stat[key]['IoU'],
                    'Dice': stat[key]['Dice'],
                    'Precision': stat[key]['Precision'],
                    'Recall': stat[key]['Recall'],
                    'F1 Score': stat[key]['F1 Score'],
                })
            else:
                entry.update({
                    'IoU': None,
                    'Dice': None,
                    'Precision': None,
                    'Recall': None,
                    'F1 Score': None,
                })
            data.append(entry)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f'Saved metrics log to {csv_path}')
