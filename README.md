# Sign Language CNN Classifier

A CNN built with PyTorch that classifies static hand gestures into 36 classes (digits 0–9, letters a–z) from grayscale images. Built for the *Machine Learning* course (SS 2026), University of Innsbruck.

**Result:** 88.07% accuracy on the held-out test set.

> Dataset not included due to size.

## What it does

Takes a $128 \times 128$ grayscale image of a hand gesture and predicts the corresponding digit or letter. The dataset (9,680 images, class-imbalanced) was split 70/15/15 into train/val/test.

## Approach

- 3-block CNN (Conv2d $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool), $1 \to 32 \to 64 \to 128$ channels, feeding into a fully connected classifier
- Data augmentation (random rotation/translation) to improve generalization
- Dropout, label smoothing, and early stopping to prevent overfitting
- Manually tuned hyperparameters, validated against a held-out set

## Tech

- Python
- PyTorch
- OpenCV

## Result

93.18% train accuracy, 88.07% test accuracy — a small enough gap to suggest the model generalizes well rather than memorizing the training set.

---
> Built with Martin Mehus. I implemented the CNN architecture and training loop.
