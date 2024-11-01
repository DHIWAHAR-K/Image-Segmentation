#utils.py
import os
import torch
import logging

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total Parameters: {total_params:,}\n')

def save_model(model, path='model_weights.pth'):
    torch.save(model.state_dict(), path)
    logging.info(f'Model saved to {path}')

def load_model(model, path='model_weights.pth'):
    model.load_state_dict(torch.load(path))
    logging.info(f'Model loaded from {path}')
    return model

if not os.path.exists("training_logs"):
    os.mkdir("training_logs")

def setup_logging(log_file='training_logs/log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )