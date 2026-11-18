# DIT5411-TsoiChungHang

Traditional Chinese Handwritten Character Recognition using TensorFlow.

## Objective
Develop a deep learning model with TensorFlow to recognize traditional Chinese handwritten characters from the AI-FREE Team dataset.

## Dataset
- Source: [Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) (full version from chenkenanalytic/handwritting_data_all)
- Total characters in dataset: 13,065
- This experiment: First 499 characters
- Split rule: First 40 samples of each character used for training, remaining samples used for testing
- Image size used: 48×48

## Data Augmentation
Augmentation was applied only on the training set using `ImageDataGenerator` with the following parameters:
- Rotation: ±15 degrees
- Width / Height shift: 0.1
- Shear: 0.1
- Zoom: 0.1

This expands the effective training samples beyond the original 40 images per class.

## Models
Two different CNN architectures were developed and compared:

| Model | Architecture | Test Accuracy |
|-------|--------------|---------------|
| model_resnet_style | Residual-style CNN with BatchNormalization + GlobalAveragePooling | **23.46%** |
| model_efficient | EfficientNet-style lightweight CNN with Swish activation | **36.26%** |

**Best model**: `model_efficient` with **36.26%** test accuracy.

## Environment
- OS: Ubuntu (WSL2)
- Python: 3.13
- Framework: TensorFlow 2.21
- GPU: NVIDIA GeForce RTX 5080

## How to Run
```bash
python prepare_data.py
python train.py
python evaluate.py

model_resnet_style: 0.2346
model_efficient:  0.3626
```

## References
Dataset: AI-FREE-Team / Traditional-Chinese-Handwriting-Dataset
Assignment: THEi DIT5411 Machine Learning
