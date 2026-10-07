# MedAdapt
## Efficient Adaptation of Vision Foundation Models for Breast Cancer Histopathology Classification


## Overview

MedAdapt is a research-oriented project investigating the use of vision foundation models and parameter-efficient adaptation techniques for medical image analysis.

The project explores whether large-scale self-supervised visual representations can be efficiently adapted to breast cancer histopathology classification under limited computational resources.


## Research Question

Can vision foundation models provide efficient and interpretable representations for medical image classification compared with conventional deep learning approaches?


## Motivation

Medical imaging AI systems often require large annotated datasets and extensive computational resources.

Recent vision foundation models have demonstrated strong transferability in computer vision tasks. This project investigates their potential for biomedical applications through efficient fine-tuning strategies.


## Objectives

- Establish a CNN-based baseline for breast histopathology classification.
- Evaluate frozen vision foundation model representations.
- Investigate parameter-efficient adaptation using lightweight adapters.
- Analyze model decisions using explainable AI techniques.


## Methodology

Pipeline:

Medical Image

↓

Vision Foundation Model Encoder

↓

Efficient Adaptation Module

↓

Classification Head

↓

Prediction + Explainability


## Experiments

The project compares:

| Approach | Description |
|---|---|
| ResNet50 | Conventional CNN baseline |
| DINOv2 Linear Probe | Frozen foundation model features |
| Adapter-based Fine-tuning | Efficient model adaptation |


## Dataset

Primary dataset:

BreakHis Breast Cancer Histopathological Database


Tasks:

- Benign vs Malignant classification


## Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC


## Explainability

Model interpretation will be investigated using Grad-CAM visualization to analyze regions influencing predictions.


## Technologies

- Python
- PyTorch
- Torchvision
- DINOv2
- Scikit-learn
- OpenCV


## Project Status

 Under active development


## Author

Zahra Alipour

Biomedical Engineering | Computer Vision | Medical AI

GitHub:
https://github.com/ZahraAlipour703