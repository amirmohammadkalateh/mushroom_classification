# mushroom_classification
````markdown
# Mushroom Classification with Neural Network

This repository contains code for building and training a neural network to classify mushrooms as edible or poisonous using the [Mushroom Classification dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle.

## Table of Contents

-   [Overview](#overview)
-   [Dataset](#dataset)
-   [Dependencies](#dependencies)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Model Architecture](#model-architecture)
-   [Results](#results)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

This project utilizes a feedforward neural network (ANN) implemented with TensorFlow and Keras to predict the edibility of mushrooms based on their features. The dataset is preprocessed using One-Hot Encoding for categorical features and Label Encoding for the target variable.

## Dataset

The dataset used is the "Mushroom Classification" dataset from Kaggle, which contains various features of mushrooms and their corresponding edibility labels (edible or poisonous).

-   **Source:** [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom) (via Kaggle)
-   **File:** `mushrooms.csv`
-   **Features:** Categorical features describing the mushroom's characteristics.
-   **Target:** `class` (edible or poisonous).

## Dependencies

-   Python 3.x
-   pandas
-   scikit-learn
-   tensorflow
-   kagglehub

You can install the required dependencies using pip:

```bash
pip install pandas scikit-learn tensorflow kagglehub
````

## Installation

1.  Clone the repository:

<!-- end list -->

```bash
git clone [repository URL]
cd [repository directory]
```

2.  Install the dependencies (see above).

## Usage

1.  Run the Python script:

<!-- end list -->

```bash
python mushroom_classification.py
```

This script will:

  - Download the dataset from Kaggle using `kagglehub`.
  - Load and preprocess the data.
  - Build and train the neural network model.
  - Evaluate the model's performance on the test set.
  - Print the test loss and accuracy.

## Model Architecture

The neural network model is built using the Keras Functional API and consists of the following layers:

  - Input layer: Shape determined by the number of features after One-Hot Encoding.
  - Dense layer: 128 units, ReLU activation.
  - Dropout layer: 20% dropout rate.
  - Dense layer: 64 units, ReLU activation.
  - Dropout layer: 20% dropout rate.
  - Output layer: 1 unit, sigmoid activation (for binary classification).

The model is compiled with the Adam optimizer and binary cross-entropy loss function.


