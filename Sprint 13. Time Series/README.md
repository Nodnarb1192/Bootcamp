# Time Series Analysis for Taxi Demand Forecasting

## Introduction

Sweet Lift Taxi company, which provides taxi services at airports, needed to forecast the number of taxi orders for the next hour in order to manage resources better, specifically to ensure adequate number of drivers during peak hours. For this task, historical data on taxi orders was available. 

## Table of Contents

1. [Data Exploration and Preprocessing](#data-exploration)
2. [Model Training and Evaluation](#model-training)
3. [Model Testing and Conclusion](#model-testing)

<a name="data-exploration"></a>
## 1. Data Exploration and Preprocessing

The dataset provided contained information on the number of orders at different timestamps. The data was resampled by one hour to prepare it for the time series forecasting task.

The data was analyzed to understand patterns and trends in taxi orders. This involved plotting the time series data, checking for stationarity, and visualizing seasonality and trends if any. 

<a name="model-training"></a>
## 2. Model Training and Evaluation

Various models were trained with different hyperparameters to forecast the number of taxi orders for the next hour. The test sample constituted 10% of the initial dataset. 

The models' performance was evaluated using the Root Mean Square Error (RMSE) metric. Based on the RMSE values, the best model was selected for further testing.

<a name="model-testing"></a>
## 3. Model Testing and Conclusion

The model that achieved the best RMSE score during training was the LightGBM algorithm. This model was then tested on the test set and achieved an RMSE score of 44.08, which was below the maximum limit of 48 set for this task.

Thus, the goal of predicting the number of taxi orders for the next hour was accomplished successfully. This model can help the Sweet Lift Taxi company to efficiently manage their resources, particularly ensuring the availability of sufficient drivers during peak hours.
