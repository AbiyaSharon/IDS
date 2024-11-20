# IDS

This project develops a **network intrusion detection system** (IDS) using a combination of deep learning models and optimization techniques to classify network traffic as either benign or part of various attack categories. The system leverages the **CICIDS 2019 dataset**, which simulates real-world network traffic with different attack scenarios.


## Project Overview

The objective of this project is to create an intrusion detection system that can accurately classify network traffic data from the **CICIDS 2019 dataset** into normal and attack classes using advanced machine learning techniques.

### Key Components:
- **Feature Extraction**: Using the **Modified Vision Transformer** to extract relevant features from raw traffic data.
- **Model Training**: Training a **Dual Attention Ghost Bidirectional Gated Recurrent Neural Network (Datt-GBiGRU)** for classification tasks.
- **Optimization**: Fine-tuning the model using the **HCBABC Optimization** algorithm for better performance.

---

## Dataset

The **CICIDS 2019** dataset includes both benign and attack traffic data, simulating real-world cyberattacks such as DDoS, brute force, and botnets.

### Features:
- **Source/Destination IP**
- **Source/Destination Port**
- **Protocol Type**
- **Packet Sizes**
- **Flow Duration**
- **Connection Statistics**
  
The dataset is split into training and test sets. The preprocessing steps include normalization and standardization to ensure all features are on a similar scale for model training.

---

## Preprocessing

### Steps:
1. **Normalization**: Uses MinMaxScaler to scale the features to a range of [0, 1].
   ```python
   X_norm = (X - min(X)) / (max(X) - min(X))
Standardization: Centers and scales the data to have zero mean and unit variance.
python
Copy code
X_std = (X - mean) / std_dev
ILA-SMOTE (Improved Linear Adaptive SMOTE) is used for balancing the dataset by generating synthetic minority class samples.

Algorithms Used
Modified Vision Transformer (ViT)
The Modified Vision Transformer (ViT) adapts the Transformer architecture, which is primarily used for images, to work with tabular data. It divides the data into patches and applies multi-head attention to learn spatial dependencies.

Patch Embedding: Divides input data into smaller patches and applies position encoding.
Multi-Head Attention: Focuses on relationships between features.
Global Average Pooling (GAP): Reduces dimensionality by summarizing the learned features into a fixed-size vector.
Why ViT?
The ViT model is useful because it can capture long-range dependencies and correlations between features that may not be immediately adjacent, which is essential for detecting anomalies in network traffic.

Datt-GBiGRU
Dual Attention Ghost Bidirectional Gated Recurrent Unit (Datt-GBiGRU) combines the power of attention mechanisms and GRU (Gated Recurrent Units) to process sequential data efficiently.

Dual Attention: Focuses on both spatial (feature) and temporal (sequence) dependencies in data.
Ghost Layers: Reduces computational overhead by generating approximate feature maps.
Bidirectional GRU: Processes the sequence in both forward and backward directions, enhancing context understanding.
Why Datt-GBiGRU?
This model is suitable for time-series data like network traffic, where understanding the sequence of events is crucial for identifying anomalies.

HCBABC Optimization
Hierarchical Cluster-Based Artificial Bee Colony (HCBABC) is used to optimize the model’s hyperparameters and select the most important features.

Phases:
Employed Bees: Search around known solutions.
Onlooker Bees: Choose solutions based on fitness.
Scout Bees: Explore new regions to avoid local optima.
Chaotic Maps: Introduces randomness into the search space to ensure better exploration.
Why HCBABC?
HCBABC is used to find the best hyperparameters and feature subsets, which significantly improves model performance and reduces overfitting.

Model Evaluation
The performance of the model is evaluated using the following metrics:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC Curve
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
The Performance.py script provides various evaluation functions that calculate these metrics.

Installation
To run this project locally, you need Python 3.6+ installed along with the following dependencies:

Requirements:
tensorflow
keras
scikit-learn
numpy
matplotlib
pandas
You can install the necessary dependencies using pip:

bash
Copy code
pip install tensorflow keras scikit-learn numpy matplotlib pandas

