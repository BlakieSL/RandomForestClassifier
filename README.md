# Random Forest Classifier Visualization

## Description
This project demonstrates how different parameters in a Random Forest Classifier affect model performance and decision boundaries. Using scikit-learn's `make_moons` dataset, it trains classifiers with varying splitting criteria (Gini impurity/entropy) and ensemble sizes, then visualizes decision boundaries alongside training/test accuracies.

## Features
- Generates a synthetic moons dataset with 10,000 samples and 0.4 noise
- Splits data into training (80%) and test (20%) sets
- Trains 8 Random Forest models with combinations of:
  - Splitting criteria: `gini` or `entropy`
  - Number of trees: `5`, `20`, `100`, `500`
- Visualizes decision boundaries and data points using Matplotlib
- Compares training/test accuracies to demonstrate ensemble learning effects
- Fixed tree depth (`max_depth=5`) to analyze impact of ensemble size

## Installation
```bash
pip install numpy matplotlib scikit-learn
