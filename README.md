# Domain Randomization for Sim-to-Real Transfer (MNIST -> SVHN)

This project explores **domain randomization** as a technique for domain adaptation. A CNN is trained on an augmented MNIST dataset (source domain) and evaluated on SVHN (target domain) — without any target domain data during training.

## Concept

Domain randomization artificially diversifies the training distribution by applying random visual augmentations to source-domain images. The hypothesis is that a model trained on sufficiently varied synthetic data will generalize to real-world data.

## Project Structure

```
.
├── train.py                        # Main training + evaluation script
├── net.py                          # CNN architectures (Net, ResNet)
├── dataset.py                      # Custom MNIST dataset with augmentation
├── domain_randomization_methods.py # Augmentation pipeline
└── data/
    └── background_images/          # Background images (Pascal VOC 2008)
```

## Domain Randomization Types

Pass `--type <int>` to select an augmentation strategy:

| Type | Description |
|------|-------------|
| 0 | No augmentation (baseline) |
| 1 | Random digit foreground color |
| 2 | Random background image |
| 3 | Random digit color + background + Gaussian noise |
| 4 | Random digit color + background + brightness/contrast |
| 5 | Random digit color + background + motion blur |
| 6 | Random digit color + background + random rotation |
| 7 | Random digit color + background + all augmentations combined |

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV (`cv2`)
- NumPy
- Pillow

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy pillow
```

## Usage

Train with a specific domain randomization type:
```bash
python train.py --type <0-7>
```

Use ResNet instead of the simple CNN:
```bash
python train.py --type 3 --use_resnet 1
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--type` | int | required | Domain randomization type (0-7) |
| `--use_resnet` | int | 0 | Set to `1` to use ResNet, `0` for simple CNN |

## Models

- **Net**: A small custom CNN (2 conv layers + 3 FC layers, 10-class output).
- **ResNet**: A ResNet-style architecture using `BasicBlock` layers from torchvision, adapted for 10-class digit classification.

## Training Details

- **Source (train)**: MNIST with domain randomization applied
- **Target (test)**: SVHN test split
- **Batch size**: 32
- **Optimizer**: SGD with momentum 0.9
- **Learning rate**: 0.001
- **Epochs**: 20
- **Loss**: Cross-entropy

## Data

Background images are from the **Pascal VOC 2008** dataset and are stored in `data/background_images/`. They are used to replace or blend with MNIST digit backgrounds during augmentation.

MNIST and SVHN datasets are downloaded automatically to `./data/` on first run.
