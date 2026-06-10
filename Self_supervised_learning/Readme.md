# Self-Supervised Contrastive Learning Across Multiple Modalities

## 📌 Project Overview
[cite_start]This repository implements a robust **Self-Supervised Contrastive Learning (SimCLR/NT-Xent)** framework tested across three distinct data modalities[cite: 28, 29]:
1. [cite_start]**MNIST:** Grayscale image data (1 channel)[cite: 29].
2. [cite_start]**CIFAR-10:** Colored image data (3 channels)[cite: 29].
3. [cite_start]**UCI-HAR:** Time-series Human Activity Recognition data[cite: 29].

[cite_start]The core framework validates that pre-training on unlabeled data captures structural patterns across different domains, drastically boosting downstream test accuracy when labeled data is scarce[cite: 42, 44, 45, 46].

---

## ⚙️ Model Architecture & Training Strategy

[cite_start]The workflow maintains a consistent backbone structure across all data types, simply adapting the input layer ("black box encoder") to fit the specific modality[cite: 30, 43]:

* [cite_start]**Data Transformations:** Every data point undergoes two distinct, domain-specific augmentations to create positive pairs[cite: 30].
* [cite_start]**Network Encoders:** * **2D-CNNs** are used to extract spatial features for MNIST and CIFAR[cite: 31].
  * [cite_start]**1D-CNNs** are used to extract temporal features for UCI-HAR[cite: 31].
  * [cite_start]**ResNet Layers:** Integrated into the CIFAR and UCI-HAR models to prevent vanishing gradient issues during deep feature extraction[cite: 32].
* [cite_start]**Loss Function:** Models are optimized using **InfoNCE / Normalized Temperature-Scaled Cross Entropy (NT-Xent) loss**[cite: 31]:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

---

## 📊 Evaluation & Key Findings

### 1. Data-Efficiency Experiment
[cite_start]We evaluated two matching network architectures under different constraint scenarios[cite: 33]:
* [cite_start]**Pre-trained Encoder + Linear Probe:** Pre-trained on all training data (unsupervised) and fine-tuned with minimal labels[cite: 34, 35].
* [cite_start]**Trained From Scratch:** Built with randomly initialized weights and trained solely on the small labeled splits[cite: 34, 35].

[cite_start]As the labeled sample sizes scale down (e.g., down to 100 sample points), the **contrastive pre-trained model holds strong with massive accuracy gains**, whereas models trained from scratch plummet[cite: 36, 39].

#### UCI-HAR Performance Curve
![UCI-HAR Pretrained vs Supervised](..images/uci_har_pretrain_vs_supervised.png)

### 2. Final Task Accuracy Summary
[cite_start]Once fully fine-tuned, the self-supervised representations achieved highly competitive results across all three benchmarks[cite: 40]:

| Dataset | Modality Type | Top Test Accuracy | Notes |
| :--- | :--- | :---: | :--- |
| **MNIST** | Grayscale Image (1-Channel) | **~97%** | [cite_start]Clean structural separation[cite: 40]. |
| **UCI-HAR** | Time-Series (1D Signal) | **~95%** | [cite_start]Robust temporal clustering[cite: 40]. |
| **CIFAR-10** | Color Image (3-Channel) | **~80%** | [cite_start]Slightly lower; pending hyperparameter tuning[cite: 40, 41]. |

---

## 🔍 Embedding Space Visualizations

[cite_start]To analyze how effectively the contrastive encoder structures its 128/256-dimensional feature representations, we applied **Principal Component Analysis (PCA)** to reduce the latent space down to 2D[cite: 37, 48, 49]. 

[cite_start]Even though collapsing complex embeddings into 2 dimensions creates natural overlapping areas, clear structural class separation and regional boundaries are easily visible[cite: 37, 38, 49].

#### CIFAR-10 Latent Space Cluster
![CIFAR Scatter Plot](..images/cifar_scatter_plot.png)

---

