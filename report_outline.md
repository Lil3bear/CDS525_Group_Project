# Report Outline

## 1. Introduction

- Project goal: classify KMNIST handwritten characters with a simple CNN.
- Reason for choosing KMNIST: small, standard image classification benchmark that fits a one-day project.

## 2. Dataset And Preprocessing

- Dataset: `torchvision.datasets.KMNIST`
- Input size: grayscale `28x28`
- Preprocessing: tensor conversion and normalization to roughly `[-1, 1]`
- Training/validation split: `90/10`

## 3. Model Choice

- Model: 2-layer CNN with max-pooling and 2 fully connected layers
- Why this model:
  easy to explain
  suitable for image classification
  fast enough for repeated experiments

## 4. Training Setup

- Main loss: cross-entropy
- Alternate loss comparison: label smoothing cross-entropy or fallback NLL loss
- Optimizer: Adam
- Baseline hyperparameters:
  learning rate `0.001`
  batch size `64`
  epochs `12`
  fixed random seed

## 5. Results

- Baseline training curves
- Loss function comparison
- Learning-rate comparison
- Batch-size comparison
- First 100 test predictions

## 6. Discussion

- Which settings performed best
- Effect of changing learning rate
- Effect of changing batch size
- Typical correct and incorrect predictions

## 7. Conclusion

- Final model performance
- Why the selected baseline is a good tradeoff between simplicity and accuracy
- Possible future improvements
