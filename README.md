# Sign Language CNN Classifier

*(Image placeholder)*

A Convolutional Neural Network built with PyTorch that translates static hand gestures into text. Built for the *Machine Learning* course (SS 2026) at the University of Innsbruck. 

> **Note**: The dataset (9,680 images) is not included in this repository due to size constraints.

## What it does

The model ingests a 128×128 grayscale image of a hand gesture and accurately predicts the corresponding character across 36 distinct classes (digits 0–9, letters a–z). 

## Approach

* **Architecture**: A 3-block CNN (Conv2d → BatchNorm → ReLU → MaxPool), scaling from 1 → 32 → 64 → 128 channels, culminating in a fully connected classifier.
* **Generalization**: Implemented data augmentation (random rotation/translation) to make the model robust.
* **Optimization**: Used dropout, label smoothing, and early stopping to prevent overfitting on the class-imbalanced dataset (split 70/15/15 into train/val/test).

## Tech Stack

* Python
* PyTorch
* OpenCV

## Results

* **Train Accuracy**: 93.18%
* **Test Accuracy**: 88.07% 

This minimal gap between training and testing accuracy suggests the model successfully generalized the gestures rather than just memorizing the training data. Success!

---

> Developed alongside *Martin Mehus*. I was responsible for implementing the CNN architecture and the training loop.