# Designing a Multimodal Foundation Model for Sleep Apnea Detection and Sleep Stage Classification

## Overview

This repository contains the work completed during the Summer Research Internship Program (SRIP) on developing a multimodal foundation model for automated sleep stage classification and sleep apnea detection.

The long-term goal of this project is to assist clinicians in analyzing overnight polysomnography (PSG) recordings by reducing the manual effort required to label sleep stages and detect apnea/hypopnea events.

The project explores Self-Supervised Learning (SSL) and Contrastive Learning techniques to learn robust representations from large amounts of unlabeled physiological data before fine-tuning on smaller labeled datasets.

---

## Motivation

Sleep stage annotation is a labor-intensive process requiring experts to label approximately 1000 epochs (30-second segments) per overnight sleep study.

In real-world healthcare settings:

- Large amounts of unlabeled physiological data are available.
- Labeled sleep datasets are expensive to obtain.
- Models trained from scratch often require significant labeled data.

Contrastive learning enables representation learning from unlabeled data and significantly improves downstream performance when only limited labeled data is available.

---

## Research Objectives

### Current Objectives

- Learn meaningful representations from unlabeled physiological signals.
- Perform sleep stage classification using self-supervised learning.
- Compare contrastive learning against traditional supervised learning.
- Evaluate performance under limited labeled data scenarios.

### Future Objectives

- Build a multimodal foundation model using:
  - EEG
  - EOG
  - EMG
  - Respiratory signals

- Fine-tune on clinical sleep datasets.

- Develop models for:
  - Sleep Stage Classification
  - Obstructive Sleep Apnea Detection
  - Hypopnea Detection

---

## Methodology

### Contrastive Learning Pipeline

For each sample:

1. Generate two augmented views.
2. Pass both views through an encoder network.
3. Learn embeddings using InfoNCE / NT-Xent loss.
4. Fine-tune using labeled data with cross-entropy loss.

```

## Contrastive Learning Pipeline

The model first learns representations from unlabeled data using contrastive learning and is later fine-tuned on labeled data for downstream classification tasks.

<p align="center">
  <img src="assets/model_architecture.png" alt="Contrastive Learning Architecture" width="700"/>
</p>

### Workflow

1. Generate two augmented views of the same sample.
2. Pass both views through a shared encoder.
3. Learn embeddings using InfoNCE / NT-Xent loss.
4. Fine-tune the pretrained encoder using labeled data.
5. Perform downstream classification.

```

Input Sample
     │
 ┌───┴───┐
 │       │
Aug1   Aug2
 │       │
Encoder Encoder
 │       │
Embedding Embedding
      │
Contrastive Loss
      │
Pretrained Encoder
      │
Fine-Tuning
      │
Classification
```

---

## Datasets Used

### 1. MNIST

- Grayscale handwritten digits
- Single channel images
- 2D CNN encoder

### 2. CIFAR-10

- RGB natural images
- 3 channels
- ResNet encoder

### 3. UCI-HAR

- Human Activity Recognition
- Multivariate time-series data
- 1D CNN + ResNet encoder

### 4. Sleep-EDF

Polysomnography (PSG) recordings containing:

- EEG Fpz-Cz
- EEG Pz-Oz
- EOG
- EMG
- Respiratory signals
- Event markers

Current experiments use:

- EEG Fpz-Cz
- EEG Pz-Oz
- EMG Submental

Each participant's signals are normalized independently before training.

---

## Model Architectures

### MNIST

- 2D CNN Encoder
- Contrastive Pretraining
- Linear Evaluation

### CIFAR-10

- ResNet Encoder
- SimCLR-style Contrastive Learning
- Fine-Tuning Classifier

### UCI-HAR

- 1D CNN + Residual Connections
- Contrastive Learning
- Activity Classification

### Sleep-EDF

- Multi-channel 1D CNN / ResNet
- Contrastive Pretraining
- Sleep Stage Classification

---

## Results

### Representation Learning Experiments

| Dataset | Accuracy |
|----------|----------|
| MNIST | ~97% |
| CIFAR-10 | ~80-85% |
| UCI-HAR | ~95-97% |

### Sleep-EDF Results

| Method | Accuracy |
|----------|----------|
| Supervised (100% Labels) | 91% |
| Contrastive Learning + Fine-Tuning | 88% |
| Supervised (10% Labels) | 86% |

These results demonstrate that contrastive pretraining achieves performance close to fully supervised training while requiring substantially fewer labeled examples.

---

## Key Findings

- Contrastive learning works effectively across multiple modalities:
  - Images
  - Multi-channel images
  - Time-series signals

- Pretrained models significantly outperform models trained from scratch when labeled data is scarce.

- Even with only a small amount of labeled data, pretrained models achieve strong performance.

- Learned embeddings exhibit meaningful clustering behavior when visualized using PCA.

---

## Repository Structure

```text
.
├── MNIST/
│   ├── training
│   ├── evaluation
│   └── visualization
│
├── CIFAR10/
│   ├── SimCLR
│   ├── training
│   └── evaluation
│
├── UCI_HAR/
│   ├── preprocessing
│   ├── training
│   └── evaluation
│
├── Sleep_EDF/
│   ├── preprocessing
│   ├── contrastive_training
│   ├── supervised_training
│   └── evaluation
│
├── notebooks/
├── results/
├── figures/
└── README.md
```

(Modify this structure to match the actual repository layout.)

---

## Installation

```bash
git clone https://github.com/GellaNagaSaiKrishna/Summer-Research---Sleep-Apnea.git

cd Summer-Research---Sleep-Apnea

pip install -r requirements.txt
```

---

## Running Experiments

### Train MNIST

```bash
python train_mnist.py
```

### Train CIFAR-10

```bash
python train_cifar.py
```

### Train UCI-HAR

```bash
python train_uci_har.py
```

### Train Sleep-EDF

```bash
python train_sleep_edf.py
```

---


## Future Work

- Multimodal representation learning using EEG, EOG, EMG and respiratory signals.
- Foundation model pretraining on large-scale sleep datasets.
- Clinical adaptation using AIIMS Delhi sleep data.
- Apnea and hypopnea event detection.
- Deployment as a clinical decision-support system.

---

## Author

**Gella Naga Sai Krishna**  
B.Tech, Computer Science and Engineering  
Indian Institute of Technology Gandhinagar

### Research Area

- Self-Supervised Learning
- Contrastive Learning
- Representation Learning
- Healthcare AI
- Sleep Analysis
- Foundation Models

---

## Acknowledgements

This work was conducted as part of the Summer Research Internship Program (SRIP) focusing on multimodal foundation models for sleep stage classification and sleep apnea detection.

---

## License

This repository is intended for academic and research purposes.
