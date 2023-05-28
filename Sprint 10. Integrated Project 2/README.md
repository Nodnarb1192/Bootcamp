# Sprint 10: Integrated Project 2

## Introduction

The project involves creating a prototype machine learning model to predict the recovery rate of gold from gold ore. The aim is to optimize the production and eliminate unprofitable parameters, leading to more profits and lesser harm to the environment.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Data Analysis](#data-analysis)
3. [Model Building](#model-building)
4. [Conclusion](#conclusion)

<a name="data-preparation"></a>
## 1. Data Preparation

The project started with opening the data files and examining the data. The correctness of recovery calculation was checked by calculating the recovery for the rougher.output.recovery feature and finding the MAE between these calculations and the feature values. The features not available in the test set were analyzed to understand their parameters and types. Finally, data preprocessing was performed to prepare the data for model training.

<a name="data-analysis"></a>
## 2. Data Analysis

In this step, the concentrations of metals (Au, Ag, Pb) at different purification stages were noted and analyzed. Also, the feed particle size distributions in the training set and in the test set were compared to ensure the validity of the model evaluation. Total concentrations of all substances at different stages (raw feed, rougher concentrate, and final concentrate) were considered to identify any abnormal values. Abnormal values were then removed from both the samples.

<a name="model-building"></a>
## 3. Model Building

A function was written to calculate the final sMAPE (Symmetric Mean Absolute Percentage Error) value. Different models were trained, and their performance was evaluated using cross-validation. The best-performing model was selected and tested on the test sample. For this project, Random Forest model was used with a hyperparameters n_estimators = 15 and max_depth = 4.

<a name="conclusion"></a>
## 4. Conclusion

The trained model outperformed the constant model with a sMAPE of 7.7%, compared to the constant model's sMAPE of 7.9%. The results indicate that the model can be effectively used to predict the recovery rate of gold from gold ore, allowing for optimized production parameters and increased profitability. The prototype machine learning model created during this project was validated and proved to be an effective tool for predicting the recovery of gold.
