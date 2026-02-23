# Landmark Classifier

A CNN-based image classification system that predicts the geographic location of a photo from 50 world landmarks. Includes a custom CNN trained from scratch and a ResNet18 transfer learning model, both exported as standalone TorchScript artifacts for deployment.

## Overview

Given a user-supplied photo, the system identifies which of 50 global landmarks (Haleakala National Park, Mount Rainier, Ljubljana Castle, Dead Sea, Temple of Olympian Zeus, etc.) appears in the image and returns the top-k predictions with confidence scores. Two model paths are compared: a 5-block CNN designed from scratch achieves 60% test accuracy, while a frozen ResNet18 backbone with a retrained classifier head reaches 72% with 100x fewer trainable parameters.

## Architecture

### CNN from Scratch

```
Input: 3x224x224

Block 1: Conv2d(3→32, 3x3) → BatchNorm → ReLU → MaxPool    → 32@112x112
Block 2: Conv2d(32→64, 3x3) → BatchNorm → ReLU → MaxPool   → 64@56x56
Block 3: Conv2d(64→128, 3x3) → BatchNorm → ReLU → MaxPool  → 128@28x28
Block 4: Conv2d(128→256, 3x3) → BatchNorm → ReLU → MaxPool → 256@14x14
Block 5: Conv2d(256→512, 3x3) → BatchNorm → ReLU           → 512@14x14

AdaptiveAvgPool2d(1,1) → Flatten → Dropout(0.7) → Linear(512→256) → ReLU → Dropout(0.7) → Linear(256→50)
```

Progressive filter expansion from 32 to 512 channels. Global average pooling replaces a large fully-connected layer, reducing parameter count and overfitting risk. Heavy dropout (0.7) compensates for the small dataset (~4,000 training images across 50 classes).

### Transfer Learning

```
ResNet18 (ImageNet pretrained, frozen) → Linear(512→50)
```

All convolutional layers are frozen — only the final 25K-parameter classifier is trained. The ImageNet backbone provides general-purpose feature extraction (edges, textures, shapes) that transfers well to landmark recognition.

### Data Pipeline

Training transforms: `Resize(256) → RandomCrop(224) → RandomHorizontalFlip → RandomRotation(15) → ColorJitter(0.2) → Normalize`

Validation/test transforms: `Resize(256) → CenterCrop(224) → Normalize`

Per-channel mean and std are computed from the training set and cached in `mean_and_std.pt`.

### Inference

The `Predictor` class wraps a trained model with its transforms into a single `nn.Module`, then exports via `torch.jit.script()` to a standalone `.pt` file. The TorchScript artifact includes all preprocessing — no source code needed to run inference.

## Results

| Model | Test Accuracy | Test Loss | Trainable Params | Convergence |
|---|---|---|---|---|
| CNN from scratch | 60% (760/1250) | 1.564 | ~2.5M | Plateaus at epoch 30 |
| ResNet18 transfer | 72% (911/1250) | 1.061 | ~25K | Plateaus at epoch 10 |

## Tech Stack

| Component | Library |
|---|---|
| Framework | PyTorch 1.11, TorchVision 0.12 |
| Training viz | livelossplot |
| Export | TorchScript (`torch.jit.script`) |
| App interface | ipywidgets (file upload + top-5 display) |

## Project Structure

```
cnn_landmark_classifier/
├── src/
│   ├── model.py          # MyModel: 5-block CNN architecture
│   ├── transfer.py       # ResNet18 transfer learning setup
│   ├── data.py           # Data loaders, augmentation, train/val split
│   ├── train.py          # Training/validation loops, ReduceLROnPlateau
│   ├── optimization.py   # Loss function, optimizer factory
│   ├── predictor.py      # TorchScript-compatible inference wrapper
│   └── helpers.py        # Dataset setup, normalization caching, plotting
├── cnn_from_scratch.ipynb    # Train custom CNN + export
├── transfer_learning.ipynb   # Train transfer model + export
├── app.ipynb                 # Interactive landmark classifier app
├── landmark_images/          # 50-class dataset (train + test)
├── checkpoints/              # Saved model weights
└── requirements.txt
```

## Setup

```bash
conda create --name landmark_clf -y python=3.7.6
conda activate landmark_clf
pip install -r requirements.txt
jupyter lab
```

Run the three notebooks in order: `cnn_from_scratch.ipynb` → `transfer_learning.ipynb` → `app.ipynb`.
