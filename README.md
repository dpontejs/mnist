# MNIST Handwritten Digit Classification with a CNN

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Test accuracy](https://img.shields.io/badge/test%20accuracy-98.7%25-success?style=flat-square)

Convolutional neural network that classifies handwritten digits (0–9) from the MNIST dataset,
reaching **98.7% accuracy on the test set**.

## Pipeline

1. **Data:** 60,000 training and 10,000 test images (28×28 grayscale) from `keras.datasets.mnist`.
2. **Preprocessing:** pixel normalization to [0, 1], channel dimension added, one-hot labels.
   The training set is split 80/20 (stratified) into train and validation.
3. **Model:** two `Conv2D` + `MaxPooling2D` blocks followed by a dense softmax classifier (Keras `Sequential`).
4. **Training:** Adam, categorical cross-entropy, batch size 128, 30 epochs, with loss/accuracy curves
   for training and validation.
5. **Evaluation:** test accuracy, classification report and confusion matrix.

## Results

| Metric | Value |
|---|---|
| Validation accuracy | 98.65% |
| **Test accuracy** | **98.69%** |
| Test loss | 0.068 |

## Running

```bash
pip install tensorflow scikit-learn matplotlib seaborn
jupyter notebook mnist.ipynb
```
