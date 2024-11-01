#main.py
import os
import torch
from model import UNet
from train import train_model
from torch.utils.data import DataLoader
from data_loader import BrainScanDataset
from utils import count_parameters, save_model, setup_logging

if __name__ == "__main__":
    setup_logging()

    # Settings for training
    train_config = {
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'n_epochs': 12,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'batches_per_epoch': 50,
        'lr_decay_factor': 1
    }

    # Print the device being used
    print(f"Using device: {train_config['device']}")

    # Directory containing .h5 files
    directory = "../image-segmentation/dataset/BraTS2020_training_data/data/"
    # Create .h5 file paths
    h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    # Split the dataset into train and validation sets (90:10)
    split_idx = int(0.9 * len(h5_files))
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:]
    # Create the train and val datasets
    train_dataset = BrainScanDataset(train_files)
    val_dataset = BrainScanDataset(val_files, deterministic=True)
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    # Create UNet model and count params
    model = UNet()
    count_parameters(model)

    # Train model
    train_epoch_losses, val_epoch_losses = train_model(model, train_dataloader, val_dataloader, train_config, verbose=True)

    # Save model
    save_model(model, 'model_weights.pth')