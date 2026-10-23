# Project 2, Week 5: Scene Classification with Transfer Learning.

## **Name:** M. Abuzar Shoukat
## **Subject:** Introduction to Applied AI (BS Electronics)
## **Lab:** Week 5, Neural Networks, Convolutions, and Transfer Learning

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

## Week 6 Proposal

**Problem:** I will build a CNN that classifies a brain MRI slice into four categories: glioma, meningioma, pituitary tumor, or no tumor. Reading MRI scans takes expert time, and a model that sorts scans or flags likely tumors could help prioritize cases for a radiologist.

**Who benefits:** Radiologists and hospitals, especially where specialists are scarce, who could use it as a second opinion or triage aid. It is a decision-support tool, not a replacement for a doctor's diagnosis.

**Dataset:** Brain Tumor MRI Dataset on Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. It has 4 classes and 7,023 images (5,712 training and 1,311 testing, about 1,300 to 1,600 per class in training). It combines the figshare, SARTAJ and Br35H datasets. License: [copy from the Kaggle page].

**Approach:** I will reuse the Week 5 pipeline: a pretrained ResNet-18, feature extraction first, then fine-tuning of the last block. Because this is medical data, I will focus on per-class recall, since missing a tumor is worse than a false alarm, and I will study which tumor types the model confuses. MRI slices differ from ImageNet photos, so I will compare how much fine-tuning helps over feature extraction alone.

## Week 6: Brain Tumor MRI Classification with ResNet-18

**Who benefits:** Radiologists and hospitals, especially where specialists are scarce, who could use the model as a second opinion or to prioritise scans for review. **Why it matters:** reading MRI scans takes expert time, and a missed tumor costs far more than a false alarm.

**Dataset:** Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle): https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset. License: [paste the License field from the Kaggle page]. 7,200 images in 4 classes (glioma, meningioma, pituitary, notumor): 5,600 training (1,400 per class) and 1,600 test (400 per class). Labels come from the source datasets (the collection combines figshare, SARTAJ and Br35H); I did not verify them.

**Data audit:** Imbalance ratio 1.0 (perfectly balanced). Images vary in size (150 to 1,000 pixels wide) and colour mode (some RGB, some grayscale), and these properties differ by class, so I treated them as a shortcut risk. 340 exact duplicate files (138 duplicate groups inside train, 15 inside test, none across). 485 of 1,600 test images (30%) have a near-duplicate in train, and the closest pairs are identical scans. File names have no patient id, so I split train and validation by MD5 group, with 0 shared groups (train 4,665, validation 935).

**Imbalance handling:** Weighted loss, weights N / (K x n_c): glioma 0.987, meningioma 0.996, notumor 1.005, pituitary 1.013 (almost no effect because the data is balanced).

**Results (test set, used once):** Macro-F1 0.905, accuracy 0.907. Recall per class: glioma 0.738 (weakest), meningioma 0.940, notumor 1.000, pituitary 0.950. Validation macro-F1 rose from 0.815 (head only) to 0.948 (after fine-tuning layer4). 40 of 1,200 tumor scans (30 glioma, 10 meningioma) were called "notumor". Glioma was most often confused with meningioma (74 images).

**Grad-CAM:** In six heatmaps (three correct, three wrong) the model looked at brain tissue and not at borders or margins, so I found no border or background shortcut. This is limited evidence: Grad-CAM is coarse and the heat was broad and central. All three wrong examples were sagittal slices, which may hint at a scan-plane cue.

**Limitations:** Public data from three merged sources, 2D slices only, unknown scanners, hospitals and patient demographics, no patient ids. The test score is probably optimistic because of the duplicate leak. **Not for clinical diagnosis or for ruling out a tumor.**

**Responsibility questions**
1. **Who could be harmed, and how?** Patients whose tumor is missed: the model called 40 of 1,200 tumor scans "no tumor", and glioma recall is only 0.738, so a patient could be reassured wrongly and treated late. Patients with false alarms (meningioma precision is 0.805) could face anxiety and unnecessary tests.
2. **Where was the model not validated?** On new hospitals, scanners and patient groups, on 3D scans or whole studies, on children, and on rare tumor types. Only one public dataset was used, 30% of its test images are copies of training images, and there is no patient-level split.
3. **What is the human role?** A radiologist reviews every scan and makes the diagnosis. The model only orders the work or offers a second opinion, and a "no tumor" prediction must never be used to dismiss a scan.
4. **What evidence would change my conclusions?** A lower score on the test images without a near-duplicate in train, a patient-level split, or an external hospital dataset; Grad-CAM showing attention on non-anatomy; or notumor recall falling after cropping the margins. Any of these would mean the 0.905 overstates real performance.
