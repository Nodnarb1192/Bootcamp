# Using Computer Vision for Age Verification in Supermarkets

## Introduction

Good Seed, a supermarket chain, is exploring the potential of Data Science to ensure their adherence to alcohol laws, specifically regarding the sale of alcohol to underage individuals. This project was undertaken to build and evaluate a model capable of verifying a person's age from a photograph.

## Table of Contents

1. [Understanding the Project](#understanding)
2. [Exploratory Data Analysis](#eda)
3. [Model Training and Evaluation](#model-training)
4. [Conclusion](#conclusion)

<a name="understanding"></a>
## 1. Understanding the Project

Before delving into the task, a quiz was conducted to verify understanding of the project statement. The task involved building a model that could verify a person's age using computer vision techniques.

<a name="eda"></a>
## 2. Exploratory Data Analysis

An exploratory data analysis was carried out to get an overall understanding of the dataset. The dataset comprised photographs of individuals with corresponding ages.

<a name="model-training"></a>
## 3. Model Training and Evaluation

The model was trained and evaluated on a GPU platform, due to the computational demands of the task. A pre-trained ResNet50 model was utilized for this purpose, and various configurations were tested, as shown in the results table.

| Dropout | Learning Rate | Test MAE |
|:-------:|---------------|:--------:|
|   None  |     0.0001    |  6.2608  |
|   0.5   |     0.0001    |  6.0046  |
|   None  |     0.0005    |  6.7832  |
|   0.5   |     0.0003    |  6.8061  |

<a name="conclusion"></a>
## 4. Conclusion

The best model configuration achieved an average error margin of 6.1 years, an impressive feat given the small size of the data. However, despite this accomplishment, an error margin of 6.1 years might not be adequate for the intended context. In terms of age verification for alcohol sales, a margin of error this large could potentially allow an underage individual to be mistakenly classified as of legal drinking age.

Several issues were identified in the data, such as variations in image resolution, lighting, face angles, race, gender, and image quality. Suggestions were made for overcoming these challenges, including adjusting image properties to be equal, training separate models for different groups, excluding non-informative pixels, and applying face restoration techniques.

While the task's accomplishment is promising, it is clear that further improvements are necessary to make the model viable for practical applications like age verification in supermarkets. Nonetheless, the project provides a promising starting point for the use of computer vision in age verification tasks.
