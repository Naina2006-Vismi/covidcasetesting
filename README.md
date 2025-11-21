COVID-19 Chest X-ray Classification using EfficientNetB0

A Deep Learning Project for Multi-class Classification, Bias Reduction & Medical Imaging Analysis

**📌Overview**
This project implements an EfficientNetB0 deep learning model to classify chest X-ray images into multiple categories such as:
	•	COVID
	•	NORMAL
	•	LUNG OPACITY
	•	VIRAL PNEUMONIA

The script includes data loading, label extraction, dataset preparation, model training, evaluation, visualization, and automatic report generation.

**📌Project Workflow-**
**1️⃣ Load Data**
	•	Loads all images from specified folders
	•	Extracts labels using filename pattern
	•	Builds a DataFrame with filename, path, and label
	•	Shows class distribution and warnings for bad files
**2️⃣ Prepare Dataset Structure**
Dataset split:
	•	Train: 70%
	•	Validation: 20%
	•	Test: 10%
**3️⃣ Data Generators**
Uses ImageDataGenerator for:
✔ Augmentation: rotation, zoom, brightness, shifting
✔ Normalization: rescale=1/255
✔ RGB support for EfficientNet
Model: EfficientNetB0

Why EfficientNetB0?
	•	Lightweight
	•	High accuracy
	•	Pretrained on ImageNet
	•	Performs well with medical imaging

Architecture Used
	•	EfficientNetB0 (imagenet pretrained, frozen)
	•	GlobalAveragePooling2D
	•	Dense → 512 → 256 → 128
	•	Dropout (0.5, 0.4, 0.3)
	•	L2 Regularization
	•	Softmax output layer
**Compiled With** - Loss : categorical_crossentropy  
Optimizer  : Adam (lr = 0.0001)  
Metrics    : accuracy, AUC, Precision, Recall

**Training**
Includes:
	•	EarlyStopping
	•	ReduceLROnPlateau
	•	ModelCheckpoint (best validation AUC)

**Model Saving**
best_model_efficientnet.keras   ← Best AUC  
final_model_efficientnet.keras  ← Last trained model

Author

Naina Vismi N
Deep Learning • Medical Imaging 
