# Intrusion Detection using Logistic Regression

This repository contains the implementation of a logistic regression-based model designed to identify unusual network traffic patterns that may indicate potential intrusions. By treating intrusion detection as a binary classification problem, this research evaluates the effectiveness of various logistic regression models and regularisation strategies in detecting intrusions.

## Features

- **Binary Classification**: The model treats intrusion detection as a binary classification problem, identifying normal and anomalous traffic patterns.
- **Regularisation Techniques**: Explores the impact of L1, L2, and Elastic Net regularisation strategies on the model's performance.
- **Dimensionality Reduction**: Investigates the advantages of using Principal Component Analysis (PCA) in combination with logistic regression to enhance model accuracy and efficiency.
- **High Accuracy**: Achieves remarkable accuracy rates, with L1 regularisation attaining an accuracy of 99.73% by carefully selecting significant features.
- **PCA and Logistic Regression**: Demonstrates the effectiveness of combining PCA with logistic regression, achieving accuracies above 99.5% using only 30 principal components.

## Getting Started

### Prerequisites

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
