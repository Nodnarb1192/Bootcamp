# Culmination Project: Predicting Customer Churn for Interconnect Telecom

## Project Overview

The final sprint aimed at applying the knowledge and skills acquired over the course of the previous sprints in a practical setting, with tasks structured to mimic real-world work environments. The main project involved developing a machine learning model prototype following specific instructions. Additionally, an extra assignment was given, which contributed to the final score. Successful completion of the sprint required attaining at least five story points (SP), units for measuring the difficulty of a task.

## Table of Contents

1. [Team Structure](#team-structure)
2. [Tasks and Scoring](#tasks)
3. [Project: Interconnect Telecom Churn Prediction](#project)
4. [Exploratory Data Analysis](#eda)
5. [Solution Approach and Challenges](#solution)
6. [Results](#results)
7. [Conclusions](#conclusions)

<a name="team-structure"></a>
## 1. Team Structure

The Student Guidance Team, acting as colleagues, played significant roles throughout the project. The team leader was responsible for delegating tasks and ensuring their successful completion, while the project reviewer reviewed the code submitted.

<a name="tasks"></a>
## 2. Tasks and Scoring

The main project was worth 4 to 6 SP, while the additional assignment was worth 1 SP. The project required crafting a work plan, conducting exploratory data analysis, developing a model, and preparing a comprehensive report.

<a name="project"></a>
## 3. Project: Interconnect Telecom Churn Prediction

Interconnect, a telecom operator, desired a model to forecast customer churn. The goal was to identify clients planning to leave so that they could be offered promotional codes and special plan options to retain them.

<a name="eda"></a>
## 4. Exploratory Data Analysis

In the data analysis phase, several insights were discovered, including:

- High churn within the first four months of subscription
- Monthly subscribers had a higher churn rate
- Fiber optic internet service users had the highest churn rates
- The churn rate has been increasing each year

<a name="solution"></a>
## 5. Solution Approach and Challenges

An in-depth exploratory analysis was performed, followed by graphing and visualization to identify seasonality and correlations. The pipeline included SMOTE and Cross-Validation scoring. Hyperparameters were tuned using Optuna. The model development phase faced various challenges such as multiple datasets, data type conversions, target type extraction, categorical features, and small, imbalanced datasets.

<a name="results"></a>
## 6. Results

Three different models were created, with one achieving promising performance metrics. The final model had a ROC AUC score of 0.85 and an accuracy of 0.78 on the training set, and a ROC AUC score of 0.81 and an accuracy of 0.75 on the test set.

<a name="conclusions"></a>
## 7. Conclusions

The final project allowed for a practical application of the knowledge and skills acquired throughout the course. Despite the encountered challenges, the task was successfully completed, and a model was developed that performed well in predicting customer churn for Interconnect Telecom. This accomplishment underscores the applicability of machine learning techniques in real-world, business-critical applications such as customer churn prediction.
