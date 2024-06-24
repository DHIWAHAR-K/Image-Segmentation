# Image Segmentation using UNet

This project implements a UNet model for segmenting brain tumors in MRI scans. It includes modules for data loading, model definition, training, and utility functions.

## Project Structure

- `data_loader.py`: Contains the `BrainScanDataset` class for loading MRI scan data.
- `model.py`: Defines the UNet model architecture.
- `train.py`: Includes functions for training the UNet model.
- `utils.py`: Contains utility functions like counting parameters, saving, and loading models.
- `main.py`: Main script to train the UNet model.

## Usage

1. **Data Preparation**: Place your MRI scan data in the `dataset` directory. Ensure the data is in the correct format for the `BrainScanDataset` class.

2. **Training**: Modify the training configuration in `main.py` and run it to train the UNet model on your data.

3. **Inference**: Use the trained model for inference by loading it with `load_model` from `utils.py` and passing images to it.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- h5py

Install dependencies using:

```bash
pip install torch numpy h5py
```

## Training

1. Adjust the training configuration in main.py according to your needs.
2. Run main.py to start training the UNet model.


## Model Evaluation
Evaluate the trained model using metrics like Dice coefficient, sensitivity, and specificity on a separate test dataset.


## License
Feel free to modify and use the code according to your needs.