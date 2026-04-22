# VGG16 CIFAR-10 Classifier

This project implements a VGG16-based image classifier for the CIFAR-10 dataset using PyTorch. It includes training, evaluation, and model management utilities, and is designed to run on CUDA-enabled GPUs.

## Features
- VGG16 architecture adapted for CIFAR-10
- Data augmentation and normalization
- Early stopping and learning rate scheduling
- Model checkpointing and loading
- Jupyter notebook for testing and inference

## Project Structure
- `main.py`: Training script
- `VGG.py`: VGG16 model definition
- `data.py`: Data loading and augmentation
- `model_utils.py`: Model saving/loading utilities
- `test_run.ipynb`: Jupyter notebook for model testing
- `requirements.txt`: Python dependencies
- `data/`: CIFAR-10 dataset files
- `test_images/`: Example images for inference

## Setup Instructions

### 1. Clone the repository
```powershell
git clone <your-repo-url>
cd VGG16
```

### 2. Create and activate a Python environment (recommended)
You can use conda or venv. Example with conda:
```powershell
conda create -n vgg16 python=3.9
conda activate vgg16
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download CIFAR-10 dataset
The dataset will be automatically downloaded to the `data/` folder when you run the code for the first time.

### 5. Train the model
```powershell
python main.py
```

### 6. Test or run inference
Open `test_run.ipynb` in VS Code or Jupyter and follow the cells to load and test the trained model.

## Notes
- CUDA is required for training. Make sure you have a CUDA-enabled GPU and the correct PyTorch version installed.
- Model checkpoints are saved in `data/saved_models/`.
- Example test images should be placed in `test_images/`.

## License
MIT License
