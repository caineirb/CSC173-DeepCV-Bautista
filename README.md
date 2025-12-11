# Deep Computer Vision for Intelligent Environmental Temperature Management
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Caine Ivan R. Bautista, 2022-0378  
**Semester:** A.Y. 2025-2026 1st Semester  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
[150-250 words: Summarize problem (e.g., "Urban waste sorting in Mindanao"), dataset, deep CV method (e.g., YOLOv8 fine-tuned on custom trash images), key results (e.g., 92% mAP), and contributions.][web:25][web:41]

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Air conditioning systems in the Philippines typically operate at fixed temperatures, leading to energy waste and user discomfort. In tropical climates like in Iligan, rooms with varying occupancy levels and diverse clothing choices often result in either overcooling or insufficient cooling. This project addresses the need for intelligent climate control by developing a computer vision system that analyzes room occupancy, human posture, and clothing type to recommend optimal temperature settings, by providing the necessary environment data to a neural network. Such a system could reduce energy consumption while improving thermal comfort in offices, classrooms, and public spaces across the region.

### Objectives
- Develop and train different deep computer vision models for detecting how many people are in a room, if they are standing or sitting, and what type of clothes they are wearing. These models will be used as the eyes of the air conditioners to know what temperature must be set in the room to make the people inside it be comfortable.
- Implement complete training pipeline including data preprocessing, model training, validation, and evaluation.
- Integrate the three models into a unified inference system that outputs temperature recommendations based on combined predictions.
- Achieve at least 85% accuracy on human detection, 80% on posture classification, and 75% on clothing classification tasks.

![Problem Demo](images/problem_example.gif) [web:41]

## Related Work
- [Paper 1: YOLOv8 for real-time detection [1]]
- [Paper 2: Transfer learning on custom datasets [2]]
- [Gap: Your unique approach, e.g., Mindanao-specific waste classes] [web:25]

## Methodology
### Dataset
- Source: 
    - Human Detection: [Human Detection Yolo](https://www.kaggle.com/datasets/hillsworld/human-detection-yolo) (27935 images)
    - Posture Classification: [Silhouettes for Human Posture Recognition](https://www.kaggle.com/datasets/mexwell/silhouettes-for-human-posture-recognition) (2,400 images)
    - Clothing Classification: [Clothing Co-Parsing Dataset](https://www.kaggle.com/datasets/balraj98/clothing-coparsing-dataset) (3102 images) + [Intelligent Multi-Layer Clothing Dataset](https://www.kaggle.com/datasets/ziya07/intelligent-multi-layer-clothing-dataset) (4,301 images)
- Split: Some datasets are already split for train/test/val, but the general split percentages are 60/20/20.
- Preprocessing: Augmentation, resizing to 640x640

### Architecture
![Model Diagram](images/architecture.png)
- Backbone: [e.g., CSPDarknet53]
- Head: [e.g., YOLO detection layers]
- Hyperparameters: Table below

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.01 |
| Epochs | 100 |
| Optimizer | SGD |

### Training Code Snippet
train.py excerpt
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640)


## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion
- Strengths: [e.g., Handles occluded trash well]
- Limitations: [e.g., Low-light performance]
- Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-YourLastName`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
```
asttokens==3.0.1
certifi==2025.11.12
charset-normalizer==3.4.4
comm==0.2.3
contourpy==1.3.3
cycler==0.12.1
debugpy==1.8.17
decorator==5.2.1
executing==2.2.1
filelock==3.20.0
fonttools==4.61.0
fsspec==2025.12.0
idna==3.11
ipykernel==7.1.0
ipython==9.8.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
Jinja2==3.1.6
jupyter_client==8.7.0
jupyter_core==5.9.1
kagglehub==0.3.13
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.7
matplotlib-inline==0.2.1
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.6.1
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
opencv-python==4.12.0.88
packaging==25.0
pandas==2.3.3
parso==0.8.5
pexpect==4.9.0
pillow==12.0.0
platformdirs==4.5.1
polars==1.36.1
polars-runtime-32==1.36.1
prompt_toolkit==3.0.52
psutil==7.1.3
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.19.2
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
pyzmq==27.1.0
requests==2.32.5
scipy==1.16.3
setuptools==80.9.0
six==1.17.0
stack-data==0.6.3
sympy==1.14.0
torch==2.9.1
torchvision==0.24.1
tornado==6.5.2
tqdm==4.67.1
traitlets==5.14.3
triton==3.5.1
typing_extensions==4.15.0
tzdata==2025.2
ultralytics==8.3.235
ultralytics-thop==2.0.18
urllib3==2.6.1
wcwidth==0.2.14

```

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

## GitHub Pages
View this project site: [https://caineirb.github.io/CSC173-DeepCV-Bautista/](https://caineirb.github.io/CSC173-DeepCV-Bautista/)
