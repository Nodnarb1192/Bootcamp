# Exploratory Data Analysis (EDA) on Instacart Data: Summary of Findings

## Project Overview

This project entailed an exploratory data analysis (EDA) on data provided by Instacart, a grocery delivery platform. The primary goal was to clean the data and derive meaningful insights into the shopping behaviors of Instacart's customers.

## Data Cleaning and Preparation

We started by cleaning and preparing the data, which involved handling missing and duplicated values. The datasets used, such as `instacart_orders.csv`, `products.csv`, `order_products.csv`, `aisles.csv`, and `departments.csv`, had some inconsistencies that we rectified for a smooth analysis.

## Findings from the EDA

### Customer Shopping Habits

- **Peak Shopping Hours**: The bulk of grocery shopping is done between 10am and 4pm, with overnight orders being quite rare.
- **Frequency of Reordering**: Typically, customers wait for about a week to place another order. However, this varies, with the average wait time being around 11 days.
- **Ordering Trends During the Week**: We observed similar order distributions on Wednesdays and Saturdays, with a slight spike in orders around noon on Saturdays.
- **Number of Orders Per Customer**: The distribution showed a rightward skew, indicating that a smaller number of customers place a large number of orders. The average number of orders per customer is 15.

### Popular Products

- **Most Ordered Products**: Bananas, organic bananas, organic strawberries, organic baby spinach, and organic hass avocado were the top five most popular products. The remaining top 20 products were a mix of fruits and vegetables.

### Additional Insights

We investigated the typical number of items per order, the top 20 most frequently reordered items, the ratio of reorders for each product, and the percentage of reorders by each customer. These findings are represented in the corresponding graphs and tables in the notebook.

## Conclusion

Our EDA provided valuable insights into the shopping habits of Instacart customers, including their preferred shopping times and products. These findings can inform strategic decision-making for Instacart.

## Future Directions

In subsequent stages of the project, we plan to delve deeper into the data using advanced statistical methods and predictive modeling. This will help us understand patterns and predict future customer behaviors. Keep an eye out for further updates!
