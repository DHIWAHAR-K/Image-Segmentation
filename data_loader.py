#data_loader.py
import h5py
import torch
import logging
import numpy as np
from torch.utils.data import Dataset

class BrainScanDataset(Dataset):
    def __init__(self, file_paths, deterministic=False):
        self.file_paths = file_paths
        if deterministic:
            np.random.seed(1)
        np.random.shuffle(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with h5py.File(file_path, 'r') as file:
                image = file['image'][()]
                mask = file['mask'][()]

                image = image.transpose((2, 0, 1))
                mask = mask.transpose((2, 0, 1))

                for i in range(image.shape[0]):
                    min_val = np.min(image[i])
                    image[i] = image[i] - min_val
                    max_val = np.max(image[i]) + 1e-4
                    image[i] = image[i] / max_val

                image = torch.tensor(image, dtype=torch.float32)
                mask = torch.tensor(mask, dtype=torch.float32)

                logging.debug(f'Loaded and processed {file_path}')

        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.file_paths))

        return image, mask