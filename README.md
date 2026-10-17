# Project 2, Week 5: Scene Classification with Transfer Learning

**Name:** M. Abuzar Shoukat
**Subject:** Introduction to Applied AI (BS Electronics)
**Lab:** Week 5, Neural Networks, Convolutions, and Transfer Learning
**Platform:** Kaggle Notebook with GPU T4

## Overview

This project builds an image classifier that recognizes six kinds of natural scenes. It uses **ResNet-18**, a CNN pretrained on ImageNet, and adapts it with transfer learning in two stages: feature extraction, then fine-tuning of the last block.

## Dataset

- **Name:** Intel Image Classification (Kaggle, uploaded by puneet6060)
- **Classes (6):** buildings, forest, glacier, mountain, sea, street
- **Size:** 14,034 training images and 3,000 test images, each about 150 x 150 pixels in RGB
- **Split used:** 11,929 train, 2,105 validation (15% of the training folder), 3,000 test

| Class | Train | Test |
|---|---|---|
| buildings | 2191 | 437 |
| forest | 2271 | 474 |
| glacier | 2404 | 553 |
| mountain | 2512 | 525 |
| sea | 2274 | 510 |
| street | 2382 | 501 |

## Method

1. **Images as tensors:** checked an image's shape, data type and value range, and reproduced a convolution by hand (output `[30, 30, 0]`) in PyTorch.
2. **Parameter counts:** a 3x3 convolution from 3 to 64 channels has 1,792 parameters. A dense layer from 67,500 inputs to 512 outputs has 34,560,512, which is about 19,000 times more. Weight sharing is what keeps convolutions small.
3. **Data pipeline:** images resized to 224 x 224 and normalized with ImageNet mean and standard deviation. Augmentation (random resized crop, horizontal flip, brightness and contrast jitter) is applied to **training data only**. Validation and test images are only resized and normalized.
4. **Feature extraction:** froze the whole ResNet-18 body and trained only a new 6-class linear layer (**3,078 trainable parameters**, about 0.03% of the network). Adam, learning rate 1e-3, 3 epochs.
5. **Fine-tuning:** unfroze `layer4` (**about 8.4 million trainable parameters**) and trained 3 more epochs with two learning rates: 1e-4 for the pretrained block and 1e-3 for the new head.
6. **Evaluation:** kept the weights from the best validation epoch, then evaluated on the test set **once**.

## Results

| Stage | Epochs | Best validation accuracy |
|---|---|---|
| Feature extraction (3,078 trainable) | 1 to 3 | 0.901 |
| Fine-tuning layer4 (about 8.4M trainable) | 4 to 6 | 0.932 |

Fine-tuning improved validation accuracy by 3.1 percentage points.

**Test accuracy (used once): 0.924** on 3,000 images.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| buildings | 0.938 | 0.892 | 0.914 |
| forest | 0.994 | 0.987 | 0.990 |
| glacier | 0.927 | 0.830 | 0.876 |
| mountain | 0.873 | 0.901 | 0.887 |
| sea | 0.912 | 0.990 | 0.949 |
| street | 0.914 | 0.952 | 0.933 |

## Error Analysis

- **Weakest classes by recall:** glacier (0.830) and buildings (0.892).
- **Most confused pair:** glacier predicted as mountain (65 images), with another 31 mountain images predicted as glacier. Both scenes contain snow, rock and sky in similar colors. This matched my prediction before training.
- **Second confusion:** buildings predicted as street (43 images) and street as buildings (23 images). Both share roads, cars and houses.
- **Most instructive error:** a house standing in a grass field under a flat grey sky was labeled "buildings" but predicted "sea" with 0.77 confidence. The model likely used the large sky and horizon as a sea cue.
- **Overfitting:** during fine-tuning, training loss kept falling (0.135 to 0.083) while validation loss rose slightly (0.227 to 0.245). This is a mild sign of overfitting, so training for more epochs would likely hurt.

## What I Learned

- Transfer learning gives high accuracy with very few trained parameters because pretrained layers already detect edges, textures and shapes.
- Training accuracy can be lower than validation accuracy early on, because augmentation makes training images harder.
- A high confidence score does not mean the prediction is right. Some errors came from images whose labels are genuinely ambiguous.

## Repository Contents

- `week-5-transfer-learning.ipynb`: full notebook with code, outputs and written answers
- `resnet18_intel.pt`: trained model weights ([size] MB)
- `README.md`: this file

## How to Run

1. Open the notebook on Kaggle and add the dataset **Intel Image Classification** (puneet6060).
2. In Settings, set Accelerator to **GPU T4** and Internet to **On**.
3. Run all cells from top to bottom. The model takes about 6 epochs of roughly 35 seconds each.

To load the saved weights:

```python
import torch, torch.nn as nn
from torchvision import models

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('resnet18_intel.pt', map_location='cpu'))
model.eval()
