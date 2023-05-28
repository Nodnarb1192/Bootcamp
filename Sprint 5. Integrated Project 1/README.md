# Integrated Project 1: Video Game Success Prediction

## Introduction

This project focuses on analyzing historical video game sales data to predict trends that could determine the success of a game. We aim to identify potential big winners and plan advertising campaigns accordingly. Our dataset contains user and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and sales data from various regions dating back to 2016.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Data Analysis](#data-analysis)
3. [User Profile per Region](#user-profile)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Conclusion](#conclusion)

<a name="data-preparation"></a>
## 1. Data Preparation

The data was initially imported and loaded into a DataFrame. The dataset contained several missing values in various columns, which were treated accordingly:

- 'name': Removed the two instances that had missing values.
- 'year_of_release': Removed a small portion of missing values and converted the remaining to integers.
- 'genre': No instances remained after cleaning 'name' and 'year_of_release' columns.
- 'critic_score' and 'user_score': Approximately one-third of the instances were missing. These were filled using the 'interpolate' function.
- 'rating': Missing values were initially filled with 'unknown'. After reviewing the data, some ratings from other systems were converted to the ESRB counterpart.

A new column 'total_sales' was added, representing the sum of sales in all regions for each game.

<a name="data-analysis"></a>
## 2. Data Analysis

Various aspects of the data were examined:

- Distribution of games released per year
- Distribution of games released per platform
- Total sales per year for each platform
- User rating correlation with total sales
- Critic rating correlation with total sales

Visual representations were created for the above analyses, including bar charts, scatter plots, boxplots, and pie charts.

<a name="user-profile"></a>
## 3. User Profile per Region

We analyzed the popularity of platforms and genres per region (NA, EU, JP). Key findings included:

- Xbox One and PS4 are the leading platforms in terms of sales.
- Action is the most popular genre, with Shooter and Sports also being popular in the NA and EU regions, and Role-Playing popular in the JP region.

We also examined the impact of ESRB ratings on sales per region, finding that they did have a significant effect.

<a name="hypothesis-testing"></a>
## 4. Hypothesis Testing

Two hypotheses were tested:

1. Average user ratings of the Xbox One and PC platforms are the same.
2. Average user ratings for the Action and Sports genres are different.

Findings:

- We rejected the null hypothesis for the first, suggesting the average user ratings for Xbox One and PC are different.
- We failed to reject the null hypothesis for the second, suggesting no significant difference in average user ratings between Action and Sports genres.

<a name="conclusion"></a>
## 5. Conclusion

Success in the video game market is dependent on the region, platform, genre, and ESRB rating. Video games will be successful if they are compatible with the Xbox One and PS4 platforms. The Action genre is predicted to be the most successful, followed by the Shooter and Sports genres in the NA and EU regions, and the Role-Playing genre in the JP region. Games rated 'Mature' are projected to be most popular in the NA and EU regions, while the most popular rating in the JP region is less clear due to a large amount of 'Rating Pending' data. 

Thus, by aligning game development and

 marketing strategies with these trends, we can increase the chances of success in the video game market.
