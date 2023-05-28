# Automated Sentiment Analysis for Movie Reviews

## Introduction

The Film Junky Union, an engaging community for classic movie enthusiasts, has embarked on a project to develop a system for filtering and categorizing movie reviews. They are interested in training a model capable of automatically identifying negative reviews. The task involved the use of an IMDb movie reviews dataset, which had been labeled according to sentiment polarity. The aim was to construct a model capable of classifying reviews as positive or negative with an F1 score of at least 0.85.

## Table of Contents

1. [Data Loading and Preprocessing](#data-loading)
2. [Exploratory Data Analysis (EDA) and Preprocessing for Modeling](#eda)
3. [Model Training and Evaluation](#model-training)
4. [Model Testing and Conclusion](#model-testing)

<a name="data-loading"></a>
## 1. Data Loading and Preprocessing

The first step involved loading the data from the 'imdb_reviews.tsv' file and preprocessing it for further analysis. Preprocessing involved techniques like removal of unwanted characters, lower casing, tokenization, and lemmatization.

<a name="eda"></a>
## 2. Exploratory Data Analysis (EDA) and Preprocessing for Modeling

EDA was conducted to understand the distribution of classes in the data, which revealed a class imbalance. This understanding was crucial as it guided the choice of metrics for model evaluation and strategies for handling the class imbalance during model training.

The data was further preprocessed to prepare it for modeling. This involved transforming the text data into numerical representations that can be understood by machine learning algorithms, using techniques such as Bag-of-Words or TF-IDF.

<a name="model-training"></a>
## 3. Model Training and Evaluation

At least three different models were trained on the dataset. Logistic Regression and Gradient Boosting models were recommended, however, the freedom to try out different methods was allowed. The evaluation of these models was done using the F1 score metric.

<a name="model-testing"></a>
## 4. Model Testing and Conclusion

The trained models were then tested on a set of newly composed reviews. Differences in the testing results of the models were observed and interpreted. 

Models' performance (F1 score) on different reviews was summarized in a table. The logistic regression model trained on the data lemmatized by spacy and DistilBERT performed relatively poorly compared to others. On the other hand, models trained on data lemmatized by NLTK performed better than those trained on data lemmatized by Spacy. 

Though the performance of the models was impressive, there's always room for improvement, possibly through more fine-tuned hyperparameter tuning. 

This project was a success as we managed to build several models capable of classifying movie reviews as positive or negative, thus fulfilling the Film Junky Union's requirements.
