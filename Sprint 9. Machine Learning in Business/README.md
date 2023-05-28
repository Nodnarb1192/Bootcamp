# Sprint 9: Machine Learning in Business

## Introduction

The project involves working for the OilyGiant mining company with the goal of finding the most profitable location for a new oil well. The process includes collecting oil well parameters, building a predictive model for volume reserves, and ultimately choosing the region that promises the highest total profit. 

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Model Training and Testing](#model-training)
3. [Profit Calculation Preparation](#profit-calculation)
4. [Profit Calculation Function](#profit-function)
5. [Risk and Profit Calculation](#risk-profit)
6. [Conclusion](#conclusion)

<a name="data-preparation"></a>
## 1. Data Preparation

In this stage, the data for each of the three regions was downloaded and prepared for further analysis. This involved processing and cleaning the data to make it suitable for model training.

<a name="model-training"></a>
## 2. Model Training and Testing

For each region, the data was split into a training set and a validation set at a ratio of 75:25. A linear regression model was trained and used to make predictions for the validation set. The average volume of predicted reserves and model RMSE were calculated and printed.

<a name="profit-calculation"></a>
## 3. Profit Calculation Preparation

In this step, all key values for calculations were stored in separate variables. The volume of reserves sufficient for developing a new well without losses was calculated and compared with the average volume of reserves in each region.

<a name="profit-function"></a>
## 4. Profit Calculation Function

A function to calculate profit from a set of selected oil wells and model predictions was written. This involved picking the wells with the highest values of predictions and summarizing the target volume of reserves in accordance with these predictions.

<a name="risk-profit"></a>
## 5. Risk and Profit Calculation

The bootstrapping technique with 1000 samples was used to find the distribution of profit. Average profit, 95% confidence interval and risk of losses were calculated. Only regions with the risk of losses lower than 2.5% were considered suitable.

<a name="conclusion"></a>
## 6. Conclusion

Taking into consideration the average profit, 95% confidence interval, and risk of losses, it was determined that Region 1 is the most suitable location for OilyGiant Mining Company to construct their new oil well. This region promises the most profitability with acceptable risk. The conclusion was made based on the predictive model for oil reserves, the calculated profit from selected wells, and the risk assessment.
