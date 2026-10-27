# Support Ticket Classification

## Project Overview

This project classifies customer support tickets into categories and priority levels using TF-IDF feature extraction and Logistic Regression. It covers dataset preparation, stratified splitting, hyperparameter tuning, model evaluation, and error analysis.

## Dataset Description

The dataset contains 600 synthetically generated support tickets. Each record includes three fields:

- `text`: The support ticket description.
- `category`: The type of issue reported.
- `priority`: The urgency level of the ticket.

### Data Source

The dataset was generated locally using Python templates for this assignment. It is a synthetic dataset and is not collected from real customers or downloaded from a public dataset repository.

### Categories

- Billing
- Technical
- Account
- Shipping
- Cancellation

### Priority Levels

- Low
- Medium
- High

### Data Cleaning

- Converted text to lowercase.
- Removed leading and trailing whitespace.
- Normalized repeated whitespace.
- Removed duplicate ticket text.
- Checked class distributions for category and priority.

### Dataset Summary

| Label | Class | Count |
|---|---|---:|
| Category | Billing | 120 |
| Category | Technical | 120 |
| Category | Account | 120 |
| Category | Shipping | 120 |
| Category | Cancellation | 120 |
| Priority | Low | 200 |
| Priority | Medium | 200 |
| Priority | High | 200 |
| **Total records** | | **600** |

### Train, Validation and Test Split

The dataset is split into training (70%), validation (15%), and test (15%) sets using stratification based on the combined category-priority labels. The random seed is fixed at 42 for reproducibility.

## Baseline Models

Two independent TF-IDF and Logistic Regression pipelines are trained:

1. Category classification
2. Priority classification

GridSearchCV with three-fold cross-validation tunes the word n-gram range and Logistic Regression regularization parameter `C`. Macro F1 is used for model selection.

## Evaluation

The program reports validation accuracy, validation macro F1, classification reports, and confusion matrices for both targets. It also evaluates the selected baseline on the held-out test set.

The bonus experiment combines word-level and character-level TF-IDF features and compares its validation performance against the word-only baseline.

## Error Analysis

The program saves up to 10 misclassified validation examples in `outputs/confusing_examples.csv`. Possible reasons include overlapping terminology, ambiguous wording, and limited variation in the synthetic training examples.

## Saved Artifacts

- `data/support_tickets.csv`: Cleaned dataset
- `data/train.csv`: Training split
- `data/validation.csv`: Validation split
- `data/test.csv`: Test split
- `models/category_model.joblib`: Category model
- `models/priority_model.joblib`: Priority model
- `outputs/dataset_summary.txt`: Class counts
- `outputs/confusion_matrix_category.png`: Category confusion matrix
- `outputs/confusion_matrix_priority.png`: Priority confusion matrix
- `outputs/confusing_examples.csv`: Error analysis

## Limitations

The dataset uses synthetic, template-based ticket descriptions. Results may overestimate performance on real-world tickets. Evaluation on a diverse, real support ticket dataset is recommended before practical deployment.

## Technologies

Python, scikit-learn, TF-IDF, Logistic Regression, GridSearchCV, Matplotlib and Joblib.
