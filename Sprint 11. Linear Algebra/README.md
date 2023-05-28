# Optimizing Insurance Predictive Modelling and Data Protection Using Linear Algebra

## Introduction

Sure Tomorrow Insurance company wanted to use machine learning to solve several of their tasks. These included identifying similar customers, predicting if a customer would receive an insurance benefit, predicting the number of insurance benefits a new customer might receive, and ensuring client data is protected. This report summarizes the findings of the project.

## Table of Contents

1. [Data Exploration and Analysis](#data-exploration)
2. [Customer Similarity](#customer-similarity)
3. [Predicting Insurance Benefits](#predicting-benefits)
4. [Data Protection](#data-protection)
5. [Conclusions](#conclusions)

<a name="data-exploration"></a>
## 1. Data Exploration and Analysis

The dataset contained information on an insured person's gender, age, salary, and the number of family members. The target variable was the number of insurance benefits received by an insured person over the last five years. The dataset contained no missing values and was appropriately formatted. The income distribution was found to be negatively skewed. 

<a name="customer-similarity"></a>
## 2. Customer Similarity

To find customers similar to a given customer, a k-Nearest Neighbors (kNN) approach was used. Data scaling was performed before implementing the kNN model as the algorithm prioritizes large values over small ones. The effect of different distance calculation methods (Manhattan and Euclidean) was explored. It was found that the distance calculation method affected the distance between points. 

<a name="predicting-benefits"></a>
## 3. Predicting Insurance Benefits

The task was to predict whether a customer would receive an insurance benefit, and if so, how many. A Linear Regression model was used for prediction. The model performed decently with an RMSE score of 0.34 and an R^2 score of 0.66.

<a name="data-protection"></a>
## 4. Data Protection

To protect client data, a data transformation algorithm was developed. The algorithm ensures that it would be hard to recover personal information if the data fell into the wrong hands (data masking or data obfuscation). The quality of machine learning models was not affected by this data transformation.

<a name="conclusions"></a>
## 5. Conclusions

The machine learning methods implemented in this project proved to be valuable tools in solving the tasks of the insurance company. They enabled the identification of similar customers, predictions of insurance benefits, and ensured data protection. It was also found that the transformation of data for protection purposes didn't affect the model's performance, demonstrating that it's possible to effectively secure client data without compromising the quality of machine learning models. The highest F1 score was 0.95 from scaled data when k was equal to 1. The obfuscated data provided the same results as the original data when used with the Linear Regression model, indicating successful implementation of data protection measures.
