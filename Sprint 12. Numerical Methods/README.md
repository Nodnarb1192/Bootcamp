# Implementing Numerical Methods for Used Car Market Value Prediction

## Introduction

Rusty Bargain, a used car sales service, is developing an application to quickly find out the market value of a car. To aid this process, a model to determine the car's value was built based on historical data including technical specifications, trim versions, and prices. 

## Table of Contents

1. [Data Exploration and Preprocessing](#data-exploration)
2. [Model Training and Evaluation](#model-training)
3. [Model Analysis](#model-analysis)
4. [Conclusions](#conclusions)

<a name="data-exploration"></a>
## 1. Data Exploration and Preprocessing

The dataset contained information on a car's technical specifications, trim versions, and prices. The features included date profile was downloaded, vehicle body type, vehicle registration year, gearbox type, power, vehicle model, mileage, vehicle registration month, fuel type, vehicle brand, whether vehicle is repaired or not, date of profile creation, number of vehicle pictures, postal code of profile owner, and date of the last activity of the user. The target variable was the car's price.

The dataset was preprocessed, which involved handling missing values, dealing with outliers, and encoding categorical features. 

<a name="model-training"></a>
## 2. Model Training and Evaluation

Four different models were trained to predict the car's price. These were Linear Regression, Decision Tree Regression, Random Forest Regression, and LightGBM. The performance of the models was evaluated using the Root Mean Square Error (RMSE) metric, and the time required for training and prediction was recorded.

The Linear Regression model had an RMSE of 3328.4 and ran in 0.1 seconds, the Decision Tree Regression model had an RMSE of 1906.2 and ran in 1.0 seconds, the Random Forest Regression model had an RMSE of 1871.8 and ran in 15.2 seconds, and the LightGBM model had an RMSE of 1575.7 and ran in 32.4 seconds.

<a name="model-analysis"></a>
## 3. Model Analysis

Among all the models, the LightGBM model outperformed others in terms of prediction accuracy, having the lowest RMSE. However, it took the longest time to run. Despite the longer runtime, the substantial reduction in RMSE made it a preferable choice.

The Random Forest model also performed reasonably well, although it had a slightly higher RMSE than the LightGBM model and required less time to run. The Decision Tree model was the fastest among the tree-based models but had a significantly higher RMSE. The Linear Regression model had the highest RMSE but was the fastest to run.

<a name="conclusions"></a>
## 4. Conclusions

In conclusion, the Light GBM model was found to be the best model for predicting a car's market value based on the historical data. It achieved the best balance between prediction accuracy (as measured by RMSE) and runtime.

However, considering the high dimensionality of the data due to numerous categorical variables, it might be beneficial to consider implementing a dimensionality reduction technique to further enhance the model's performance. It's always a trade-off between the time taken and accuracy. In real-world scenarios, depending upon the use case, we might prefer a slightly less accurate but faster model or vice versa.
