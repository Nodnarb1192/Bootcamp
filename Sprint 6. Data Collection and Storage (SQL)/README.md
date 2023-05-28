# Integrated Project 2: Data Collection and Storage

## Introduction

This project involves analyzing data for Zuber, a new ride-sharing company that's launching in Chicago. Our aim is to understand passenger preferences and the impact of external factors such as weather on ride frequency. We will be parsing weather data, analyzing taxi ride data, and testing a hypothesis about the impact of weather on ride frequency.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Exploratory Data Analysis](#data-analysis)
3. [Hypothesis Testing](#hypothesis-testing)
4. [Conclusion](#conclusion)

<a name="data-collection"></a>
## 1. Data Collection

Data was parsed from the website providing weather information for Chicago in November 2017. The information was then organized into a database with various tables containing information on city neighborhoods, taxis, rides, and weather records. 

<a name="data-analysis"></a>
## 2. Exploratory Data Analysis

In the exploratory data analysis stage, we addressed several questions:

- The number of taxi rides for each taxi company for November 15-16, 2017.
- The number of rides for every taxi company whose name contains the words "Yellow" or "Blue" for November 1-7, 2017.
- The number of rides for Flash Cab and Taxi Affiliation Services, the most popular taxi companies in November 2017.

Visualizations such as bar charts and pie charts were used to depict the findings.

<a name="hypothesis-testing"></a>
## 3. Hypothesis Testing

We tested the hypothesis that the duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays. For this, we retrieved identifiers of the O'Hare and Loop neighborhoods from the neighborhoods table and weather condition records from the weather_records table. Then, we categorized all hours into two groups: "Bad" if the description field contains the words "rain" or "storm," and "Good" for others. 

Rides data was then retrieved from the trips table for all the rides that started in the Loop and ended at O'Hare on a Saturday. We ensured to get the weather conditions for each ride and also the duration of each ride. 

<a name="conclusion"></a>
## 4. Conclusion

Based on the exploratory data analysis, we identified trends and patterns in taxi rides in Chicago. Our hypothesis testing helped us understand the impact of weather on ride frequency, particularly from the Loop to O'Hare International Airport on rainy Saturdays.

The insights and findings from this project will provide Zuber, the ride-sharing company, with a clear understanding of passenger preferences and the impact of external factors on ride frequency. This information will be crucial in shaping their operations and marketing strategies as they launch in Chicago.
