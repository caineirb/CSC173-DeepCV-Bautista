# CSC173 Deep Computer Vision Project Progress Report
**Student:** Caine Ivan R. Bautista, 2022-0378
**Date:** December 11, 2025
**Repository:** [CSC173-DeepCV-Bautista](https://github.com/caineirb/CSC173-DeepCV-Bautista)  
**Commits Since Proposal:** 3 commits | **Last Commit:** December 11, 2025

## 📊 Current Status
|      **Milestone**      |       **Human Detection**       |        **Posture Classification**       |       **Clothing Classification**       |
| :---------------------: | :---------------------------------------: | :-------------------------------------: | :-------------------------------------: |
| **Dataset Preparation** |✅ Completed|⏳ Pending<br>Scheduled Tomorrow|⏳ Not Started<br>Next Phase|
|   **Initial Training**  |✅ Completed|⏳ Not Started<br>Scheduled tomorrow|⏳ Not Started<br>Next Phase|
| **Baseline Evaluation** |✅ Completed|⏳ Not Started<br>Scheduled after training|⏳ Not Started<br>Next Phase|
|  **Model Fine-tuning**  |⏳ Not Started<br>Scheduled after initial models training |⏳ Not Started<br>Scheduled after initial models training |⏳ Not Started<br>Scheduled after initial models training |

## 1. Dataset Progress
- **Total images:** 
    - Human Detection: 27,935
- **Train/Val/Test split:**
    - Human Detection: 71.79%/3.94%/24.27% or 20,054/1,100/6,781
- **Classes implemented:**
    - Human Detection: ["Person"]
- **Preprocessing applied:** Resize(640x640), augmentation (flip, rotate, brightness)

**Sample data preview:**
### Human Detection
![Human Detection Sample](images/human_detection_sample.jpg)

## 2. Training Progress
**Training Curves (so far)**
### Human Detection
![Human Detection Results](images/human_detection_results.png)

**Current Metrics:**
### Human Detection
| Metric | Train | Val |
|--------|-------|-----|
| box_loss | 1.35 | 1.34 |
| cls_loss | 1.13 | 1.11 |
| dfl_loss | 1.37 | 1.35 |
| mAP@0.5 | 95.19% | 95.19% |
| mAP50-95 | 79.82% | 79.82% |
| Precision | 1.0 | 1.0 |
| Recall | 0.90 | 0.90 |


## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| CUDA out of memory | ✅ Fixed | Reduced batch_size from 32→16 and workers set to 2 |

## 4. Next Steps (Before Final Submission)
- [ ] Complete training (50 more epochs)
- [ ] Hyperparameter tuning (learning rate, augmentations)
- [ ] Baseline comparison (vs. original pre-trained model)
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results