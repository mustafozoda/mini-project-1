# Census Income Prediction Project

## Project Overview

In this project, we aim to predict whether an individual’s annual income exceeds $50,000 based on various demographic and employment-related features. The project is part of a competition where the first phase restricts the model to only using a **Random Forest Classifier**, while the second phase allows the use of any model.

## Data

The dataset is derived from the UCI Adult Census Income dataset and consists of the following files:

- **train.csv**: Contains the training data with features and labels (income column).
- **test.csv**: Contains test data with features (without the label) for making predictions.

The input data includes various demographic and employment-related features such as age, education, and hours worked per week.

## Competition Rules

- **First Competition**: Only **Random Forest Classifier** is allowed.
- **Second Competition**: Any model can be used.

The models are evaluated using **balanced accuracy**, which accounts for both True Positive and True Negative rates.

## Modeling Approach

### 1. **First Competition Model (Random Forest)**

- A **Random Forest Classifier** was trained using the training dataset (`train.csv`).
- Hyperparameters were fine-tuned using **Grid Search** to optimize model performance.
- Performance was evaluated on the test set, and predictions were made in accordance with the rules of the competition.

### 2. **Second Competition Model (Stacking Classifier)**

- A **Stacking Classifier** was implemented combining different base models, such as **Logistic Regression, Random Forest, and Gradient Boosting**, to predict the target variable.
- The stacking method was chosen for its ability to improve predictive performance by combining multiple models.
- We used **cross-validation** and **hyperparameter tuning** to enhance the final model.

## Evaluation Metric

The models were evaluated using **balanced accuracy**, which calculates the average of the True Positive Rate (TPR) and True Negative Rate (TNR).

## How to Run the Code

1. Ensure you have the necessary Python libraries installed (e.g., `pandas`, `sklearn`, `matplotlib`).
2. Load the dataset (`train.csv` and `test.csv`).
3. Follow the steps in the provided Jupyter notebook to train the model and make predictions.
4. Export your predictions and submit the CSV file in the required format.

## Requirements

- Python 3.7 or higher
- Required libraries:

  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

  Install required libraries using:

  ```bash
  pip install -r requirements.txt
  ```
