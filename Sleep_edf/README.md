## Method & Evaluation

### 🔌 Input Channels
We restrict our model training to an optimized 3-channel setup:
* **EEG-Fpz-Cz** (Electroencephalogram - Frontopolar-Central)
* **EEG-Pz-Oz** (Electroencephalogram - Parietal-Occipital)
* **EMG-submental** (Electromyogram - Chin)

### ⚙️ Pipeline & Training Strategy
1. **Subject-Specific Normalization:** To account for varying natural baselines across participants, data is normalized separately per individual prior to applying transformations.
2. **Semi-Supervised Fine-Tuning:** We deploy the same core architecture as our baseline experiments, evaluating the model using only **10% labeled data** after contrastive pre-training.

### 📊 Results & Comparative Analysis

We evaluated the performance of our contrastive learning model against both a fully supervised upper bound and a data-constrained supervised competitor.

| Model Approach | Labeled Data Used | Accuracy | Evaluation Type |
| :--- | :---: | :---: | :--- |
| **Supervised Baseline** | 100% | **91%** | Upper-Bound Benchmark |
| **Contrastive Learning (Ours)** | 10% | **88%** | Proposed Method |
| **Supervised Competitor** | 10% | **86%** | Low-Data Control |

> **Note on Performance:** Although the numerical variance between the 10% supervised model and our approach is subtle, the 2% improvement consistently demonstrates a distinct **contrastive learning effect**, validating the benefit of self-supervised representations in low-resource settings.

### 🔍 Latent Space Visualization

To visually verify the contrastive learning effect, we extracted the latent space embeddings of the test set and plotted them using Principal Component Analysis (PCA). 

![PCA of Sleep Stage Embeddings](images/sleep_edf_scatter_plot.png)

The scatter plot highlights how our 3-channel representation framework successfully organizes the feature space across different physiological stages (Wake, N1, N2, N3, REM), showing clear structural clustering even with limited downstream supervision.
