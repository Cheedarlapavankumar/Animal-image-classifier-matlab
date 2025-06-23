# Animal Image Classifier using Transfer Learning (MATLAB - ResNet-18)

This project implements an image classification model using transfer learning on the ResNet-18 architecture in MATLAB. The model is trained to classify images of three animal categories: dog (`cane`), elephant (`elefante`), and butterfly (`farfalla`). The goal was to achieve over 90% test accuracy within a limited training time using a well-optimized training strategy.

---

## Table of Contents

- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training Workflow](#training-workflow)
- [Evaluation](#evaluation)
- [Instructions to Run](#instructions-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Technologies Used

- MATLAB R2023a or higher
- Deep Learning Toolbox
- Pretrained ResNet-18 Network
- ImageDatastore and AugmentedImageDatastore
- Transfer Learning & Data Augmentation
- Performance evaluation metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score, ROC Curve

---

## Project Structure

Animal-Image-Classifier/
├── train_model_Classification using Transfer Learning (ResNet-18) in MATLAB.m
├── view_results.m
├── SamplePredictions/
│ ├── SamplePredictions_1.png
│ ├── SamplePredictions_2.png
│ └── ...
├── README.md
├── .gitignore
└── ClassificationResults.mat (ignored)

---

## Dataset

- Folder structure: `raw-img/animal-name/*.jpg`
- Dataset used: Custom dataset of 3 classes: `cane`, `elefante`, `farfalla`
- Balanced: Each class was limited to the smallest class size (1445 images per class)
- Split:
  - 75% training
  - 12.5% validation
  - 12.5% testing

> Note: The dataset folder is excluded from the repository via `.gitignore`.

---

## Training Workflow

- Model: ResNet-18 (pretrained on ImageNet)
- Final classification layer replaced with a new fully connected layer with 3 outputs
- Dropout added to reduce overfitting
- Data Augmentation used:
  - Rotation, Translation, Shear, Scaling, X-Reflection
- Optimizer: Adam
- Epochs: 20
- Learning Rate Scheduling applied

---

## Evaluation

- Evaluation script (`view_results.m`) loads the trained model and metrics from `ClassificationResults.mat`
- Metrics computed:
  - Overall Test and Validation Accuracy
  - Per-class Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score (per class)
  - ROC Curve (One-vs-All)
  - Bar chart of per-class metrics

---

## Instructions to Run

1. Place your dataset in `C:/your_dataset_folder/raw-img` with subfolders: `cane`, `elefante`, `farfalla`.
2. Run the training script:

```matlab
train_model_Classification using Transfer Learning (ResNet-18) in MATLAB.m
3. Once training is complete, run the evaluation script:
view_results.m
4. Check output folders for saved predictions, metrics, and plots.
Results
Test Accuracy Achieved: ~95.74%

Training Time: ~84 minutes on CPU

Target Accuracy (90%) achieved

Confusion matrix and class-wise prediction performance visualized

All plots (confusion matrix, ROC, bar charts) saved under /SamplePredictions/

License
This project is for educational purposes. Please cite if reused or adapted.
