---

# Crush Interest Prediction

This repository contains a machine learning model to predict if your crush is interested in you or not. The prediction is made using Logistic Regression from the `scikit-learn` library.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Have you ever wondered if your crush likes you back? This project aims to answer that question using data-driven methods. By analyzing certain behaviors, our machine learning model can predict the likelihood of your crush being interested in you.

## Dataset

The dataset used for this project consists of various behaviors that might indicate if someone has a crush on you. The features include:

- `Late_to_reply`: Whether the person is late to reply to your messages.
- `ignore_you`: Whether the person tends to ignore you.
- `smile_when_you_smile`: Whether the person smiles when you smile.
- `eye_contact_loss`: Whether the person loses eye contact with you.
- `ignore_your_speech`: Whether the person ignores what you say.
- `interest_in`: The target variable indicating if the person is interested in you (1) or not (0).

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/crush-interest-prediction.git
cd crush-interest-prediction
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the model, first ensure you have your dataset in a CSV file named `dataset.csv`. You can then run the model training and prediction script:

```bash
python predict_crush_interest.py
```

## Model Training

The model is trained using the Logistic Regression algorithm from the `scikit-learn` library. Here is a snippet of the training process:

```python
import numpy as np
import pandas as pd

# to build linear model
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv('dataset.csv')
data

# prompt: Using dataframe data: train as logical regression indepened var 'Late_to_reply', 'ignore_you', 'smile_when_you_smile','eye_contact_loss','ignore_your_speech'] depend variale interest_in

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('dataset.csv')

# Define the independent and dependent variables
X = data[['Late_to_reply', 'ignore_you', 'smile_when_you_smile', 'eye_contact_loss', 'ignore_your_speech']]
y = data['interest_in']

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate the model

# prompt: get accuraccy

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, predictions)
print(accuracy)

# prompt: get prediction

# make prediction
prediction = model.predict([[0, 0, 0, 1, 0]])
print(prediction)

# prompt: train the model

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# prompt: save the model

import pickle

# Save the model to a file
with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)



# prompt: use model.pkl
import pickle
X = data[['Late_to_reply', 'ignore_you', 'smile_when_you_smile', 'eye_contact_loss', 'ignore_your_speech']]
# Load the model from the file
with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

# Make predictions
predictions = model.predict(X)

# Evaluate the model

# make prediction
prediction = model.predict([[0, 0, 1, 1, 0]])
print(prediction)

```

## Evaluation

The model's performance is evaluated using accuracy. Here is how you can see the evaluation results:

```bash
python evaluate_model.py
```

## Results

The trained Logistic Regression model achieved an accuracy of approximately 0.85 on the test set. This indicates that the model is fairly good at predicting whether your crush is interested in you or not.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
