# CSC173 Deep Computer Vision Project Proposal
**Student:** Caine Ivan R. Bautista, 2022-0378  
**Date:** December 11, 2025

## 1. Project Title 
Deep Computer Vision for Intelligent Indoor Temperature Management

## 2. Problem Statement
Air conditioning systems in the Philippines typically operate at fixed temperatures, leading to energy waste and user discomfort. In tropical climates like in Iligan, rooms with varying occupancy levels and diverse clothing choices often result in either overcooling or insufficient cooling. This project addresses the need for intelligent climate control by developing a computer vision system that analyzes room occupancy, human posture, and clothing type to recommend optimal temperature settings, by providing the necessary environment data to a neural network. Such a system could reduce energy consumption while improving thermal comfort in offices, classrooms, and public spaces across the region.

## 3. Objectives
- Develop and train different deep computer vision models for detecting how many people are in a room, if they are standing or sitting, and what type of clothes they are wearing. These models will be used as the eyes of the air conditioners to know what temperature must be set in the room to make the people inside it be comfortable.
- Implement complete training pipeline including data preprocessing, model training, validation, and evaluation.
- Integrate the three models into a unified inference system that outputs temperature recommendations based on combined predictions.
- Achieve at least 85% accuracy on human detection, 80% on posture classification, and 75% on clothing classification tasks.

## 4. Dataset Plan
- Source: 
    - Human Detection: [Human Detection Yolo](https://www.kaggle.com/datasets/hillsworld/human-detection-yolo) (27935 images)
    - Posture Classification: [Silhouettes for Human Posture Recognition](https://www.kaggle.com/datasets/mexwell/silhouettes-for-human-posture-recognition) (2,400 images)
    - Clothing Classification: [Clothes Dataset](https://www.kaggle.com/datasets/ryanbadai/clothes-dataset) (7,500 images)
- Classes: 
    - Human Detection: ['Person']
    - Posture Classification: ['Sitting', 'Standing']
    - Clothing Classification: ['Light clothing' (t-shirt, shorts, dress), 'Medium clothing' (long sleeves, pants), 'Heavy clothing' (jacket, sweater, layered)]
- Acquisition: Downloaded from Kaggle Public Datasets

## 5. Technical Approach
- Architecture sketch:
```
  Input Image → Human Detection (YOLO11n) → Person Count + Bounding Boxes
                                          ↓
                              Crop detected persons
                                          ↓
                              ┌───────────┴───────────┐
                              ↓                       ↓
                   Posture Classifier        Clothing Classifier
                     (ResNet-18)                  (ResNet-50)
                              ↓                       ↓
                   Sitting/Standing          Light/Medium/Heavy
                              └───────────┬───────────┘
                                          ↓
                              Temperature Recommendation Logic
```
- Models: 
    - Human Detection: YOLO11n
    - Posture Classification: ResNet-18
    - Clothing Classification: ResNet-50
- Framework: PyTorch with torchvision for model implementations
- Training Strategy: Transfer learning and fine-tuning pre-trained models
- Hardware: Local GPU (CUDA enabled)

## 6. Expected Challenges & Mitigations
> - Challenge: Occlusion and overlapping persons may reduce detection accuracy and affect person counting
> - Solution: Use YOLO's multi-scale detection and non-maximum suppression; augment training data with crowded scenes; implement confidence thresholding
> - Challenge: Posture classification may be difficult with partial body visibility or unusual poses
> - Solution: Focus on upper body features in cropped images; augment dataset with synthetic poses; use ensemble predictions if confidence is low
> - Challenge: Limited labeled data for clothing thermal properties
> - Solution: Manually relabel subsets of clothing datasets based on material thickness