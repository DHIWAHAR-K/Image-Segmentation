# train.py
import torch
from torch.utils.data import DataLoader
from model import UNet
from data_loader import BrainScanDataset

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params:,}\n')

def save_model(model, path='model_weights.pth'):
    torch.save(model.state_dict(), path)

def train_model(model, train_dataloader, val_dataloader, config, verbose=True):
    device = config['device']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    batches_per_epoch = config['batches_per_epoch']
    lr_decay_factor = config['lr_decay_factor']

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_epoch_losses = []
    val_epoch_losses = []
    
    print("Training...")
    for epoch in range(1, n_epochs + 1):
        current_lr = learning_rate * (lr_decay_factor ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        train_epoch_loss = 0
        for train_batch_idx, (train_inputs, train_targets) in enumerate(train_dataloader, start=1):
            if verbose: print(f"\rTrain batch: {train_batch_idx}/{batches_per_epoch}, Avg batch loss: {train_epoch_loss/train_batch_idx:.6f}", end='')
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
            train_preds = model(train_inputs)
            train_batch_loss = loss_fn(train_preds, train_targets)
            train_epoch_loss += train_batch_loss.item()

            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

            if train_batch_idx >= batches_per_epoch:
                if verbose: print()
                break
        train_epoch_losses.append(train_epoch_loss)

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(val_dataloader, start=1):
                if verbose: print(f"\rVal batch: {val_batch_idx}/{batches_per_epoch}, Avg batch loss: {val_epoch_loss/val_batch_idx:.6f}", end='')
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_preds = model(val_inputs)
                val_batch_loss = loss_fn(val_preds, val_targets)
                val_epoch_loss += val_batch_loss.item()
                
                if val_batch_idx >= batches_per_epoch:
                    if verbose: print()
                    break
        val_epoch_losses.append(val_epoch_loss)

        if verbose: print(f"Epoch: {epoch}, Train loss: {train_epoch_loss:.6f}, Val loss: {val_epoch_loss:.6f}, lr {current_lr:.6f}\n")
        
    print("Training complete.")
    return train_epoch_losses, val_epoch_losses

if __name__ == "__main__":
    # Settings for training
    train_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 12,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'batches_per_epoch': 50,
        'lr_decay_factor': 1
    }

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
