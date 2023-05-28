# Sprint 7: Introduction to Machine Learning

## Introduction

The goal of this project is to develop a model that analyzes subscribers' behavior for a mobile carrier, Megaline, to recommend one of their newer plans: Smart or Ultra. The aim is to transition users from legacy plans to the new ones. The model's accuracy needs to exceed a threshold of 0.75. The data available for this project is monthly behavior information of subscribers who already switched to new plans.

## Table of Contents

1. [Data Loading and Exploration](#data-loading)
2. [Data Preparation](#data-preparation)
3. [Model Development](#model-development)
4. [Model Testing](#model-testing)
5. [Conclusion](#conclusion)

<a name="data-loading"></a>
## 1. Data Loading and Exploration

In this stage, the data file `datasets/users_behavior.csv` was opened and explored. The file contains information on:
- calls — number of calls
- minutes — total call duration in minutes
- messages — number of text messages
- mb_used — Internet traffic used in MB
- is_ultra — plan for the current month (Ultra - 1, Smart - 0)

<a name="data-preparation"></a>
## 2. Data Preparation

The data was then split into a training set, a validation set, and a test set. The training set consisted of 60% of the data, while the validation and test set each consisted of 20% of the data. 

<a name="model-development"></a>
## 3. Model Development

The quality of different models was investigated by changing hyperparameters. Two models, a decision tree model and a random forest model, were trained and the one with the highest accuracy was chosen for further testing. 

<a name="model-testing"></a>
## 4. Model Testing

The chosen model was tested using the test dataset to check the quality and accuracy of the model. The model was found to exceed the 75% accuracy threshold set by Megaline. 

<a name="conclusion"></a>
## 5. Conclusion

The accuracy of the best decision tree model was 78.54%, and the accuracy of the best random forest model was 80.56%. Both models surpassed the required accuracy threshold of 75%, but the random forest model had a slightly higher accuracy than the decision tree model.

This machine learning model will be useful to Megaline to recommend the most suitable plan to their customers who are currently using legacy plans. This will help increase customer satisfaction, reduce churn rate, and potentially increase revenue for the company.
