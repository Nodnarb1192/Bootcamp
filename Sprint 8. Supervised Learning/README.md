# Delving into Supervised Learning: Concepts and Applications

## Introduction

Beta Bank is facing a customer churn problem. The aim of this project is to build a predictive model to identify customers who are likely to churn. This can allow the bank to proactively address their issues and improve retention. The model needs to have an F1 score of at least 0.59. Additionally, we will measure the AUC-ROC metric and compare it with the F1.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Exploratory Data Analysis](#eda)
3. [Model Training without Balancing Classes](#unbalanced-training)
4. [Balancing Classes and Model Training](#balanced-training)
5. [Final Model Testing](#model-testing)
6. [Conclusion](#conclusion)

<a name="data-preparation"></a>
## 1. Data Preparation

In this stage, the data file `datasets/Churn.csv` was opened and prepared. The dataset contains customer behavior and contract termination details. The data was preprocessed, transformed into numeric form using One-Hot Encoding for categorical features, and all features were standardized to ensure equal importance.

<a name="eda"></a>
## 2. Exploratory Data Analysis

Upon examining the balance of classes, it was noticed that there was an imbalance, with the negative class accounting for around 80% and the positive class accounting for only 20% of the data.

<a name="unbalanced-training"></a>
## 3. Model Training without Balancing Classes

Logistic Regression model was initially trained and tested without taking into account the imbalance of classes. 

<a name="balanced-training"></a>
## 4. Balancing Classes and Model Training

Approaches to fixing the class imbalance were implemented for further model training. Logistic Regression, Decision Tree, and Random Forest models were trained with class_weight='balanced' parameter, upsampled data, and downsampled data. 

<a name="model-testing"></a>
## 5. Final Model Testing

The best performing model, which was the Random Forest Classifier trained with upsampled data, using a max depth of 14 and n_estimators of 20, was selected for final testing. The F1 score of the model was 0.5931, exceeding the minimum score requirement set by Beta Bank.

<a name="conclusion"></a>
## 6. Conclusion

The best performing model was able to predict customer churn with an F1 score that meets Beta Bank's requirements. Therefore, this model can be used to predict whether a customer will churn and allow the bank to take proactive measures to retain customers, leading to cost savings in the long run.
