# FGSM: Fast Gradient Sign Method Experiments

This repository contains code and scripts for running and reproducing experiments with the Fast Gradient Sign Method (FGSM) on CIFAR-10 and MNIST datasets, using ResNet18 and ViT-T/16 architectures.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies (see `requirements.txt` if available)

## Getting Started

### 1. Download Checkpoints

Before running attacks, download the pretrained checkpoints and place them in the project root folder to reproduce the reported results:
```
# Install gdown if not already installed
pip install gdown

# Download the pretrained checkpoints from Google Drive
gdown https://drive.google.com/uc?id=1jbuI-dWhU93UzaNQpu1YjVro4wke0OSl

# Extract the archive into the project root folder
unzip filename.zip -d ./
```

### 2. Training

If you prefer, train the models from scratch using the following commands:

```bash
# Train ResNet18 on CIFAR-10
bash scripts/train/resnet18_cifar10.sh

# Train ResNet18 on MNIST
bash scripts/train/resnet18_mnist.sh

# Train Tiny-ViT on CIFAR-10
bash scripts/train/tiny_vit_cifar10.sh

# Train Tiny-ViT on MNIST
bash scripts/train/tiny_vit_mnist.sh
```

### 3. Attacking

To run FGSM attacks on the trained models, use the following commands:

```bash
# Attack ResNet18 on CIFAR-10
bash scripts/attack/resnet18_cifar10.sh

# Attack ResNet18 on MNIST
bash scripts/attack/resnet18_mnist.sh

# Attack Tiny-ViT on CIFAR-10
bash scripts/attack/tiny_vit_cifar10.sh

# Attack Tiny-ViT on MNIST
bash scripts/attack/tiny_vit_mnist.sh
```

### 4. Results

- **Logs:** All results and logs will be stored in the `logs/` directory.
- **Visualizations:** Visual outputs and figures will be saved in the `assets/` directory.

---

For any questions or issues, please open an issue in this repository.
