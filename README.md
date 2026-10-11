# Anomaly-Aware Brain Tumor Segmentation (BraTS 2021)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Medical Imaging](https://img.shields.io/badge/Medical_Imaging-NIfTI-blue?style=for-the-badge)

##  Project Overview
This project presents an advanced hybrid deep learning architecture for accurately segmenting brain tumors from multi-modal MRI scans (T1, T1ce, T2, FLAIR). By leveraging the **BraTS 2021 dataset**, the model is specifically designed to handle severe class imbalances and subtle tumor boundaries.

Instead of a traditional U-Net baseline, this solution introduces a highly optimized pipeline featuring a **Symmetry Refinement Network (SRN)** and **Attention Gates**, pushing the segmentation performance to academic standards.

## Model Architecture & Key Innovations

*   **Hybrid SEGNET Backbone:** Incorporates Convolutional layers with `LightTransformer` modules to capture both local textures and global context effectively.
*   **Symmetry Refinement Network (SRN):** An intelligent module that generates pseudo-healthy brain slices by exploiting brain symmetry. This acts as a powerful prior to highlight abnormal regions.
*   **Attention Gates:** Forces the model to focus specifically on tumor cores (necrosis, edema, enhancing structures) while suppressing irrelevant background noise.
*   **Residual Anomaly Module:** Calculates pixel-wise anomaly scores to selectively boost non-background logits during inference, resulting in highly precise edge detection.
*   **Custom Loss Function (`DiceCELoss`):** A weighted combination of Cross-Entropy and Dice Loss, strictly formulated to penalize false negatives in critical enhancing tumor regions.

## Performance & Evaluation
The model was evaluated against standard BraTS metrics, demonstrating state-of-the-art competitive performance, particularly outperforming standard IntroVAE baselines in Whole Tumor detection.

| BraTS Region | Metric | Score |
| :--- | :--- | :--- |
| **Whole Tumor (WT)** | Mean Dice | **0.9800** |
| **Tumor Core (TC)** | Mean Dice | **0.8287** |
| **Enhancing Tumor (ET)** | Mean Dice | **0.8102** |

## Tech Stack & Tools
*   **Frameworks:** PyTorch, Scikit-Learn
*   **Data Processing:** NiBabel (for `.nii.gz` 3D volumes), SciPy, NumPy
*   **Environment:** Optimized for GPU execution (CUDA) with Multiprocessing pipelines for rapid data extraction.

##  Key Takeaways
This project demonstrates expertise in:
1.  Handling complex, multi-channel 3D medical imaging data.
2.  Designing custom neural network architectures beyond standard library implementations.
3.  Solving real-world data imbalance issues through algorithmic innovation (SRN & Anomaly Boosting).
