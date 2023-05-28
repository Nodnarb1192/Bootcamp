# Statistical Data Analysis for Megaline Telecom: Evaluating Revenue from Prepaid Plans

## Initial Steps

The project started with the integration of necessary Python libraries for data analysis and the importing of various CSV files into respective DataFrame structures. The primary goal was to refine and prepare the data for an insightful statistical analysis.

## Data Preparation

This phase involved multiple steps, including:

1. Converting date strings to datetime objects
2. Adjusting the internet usage limits for Surf and Ultimate plans
3. Creating columns for numeric and string values, signifying the month of each message, call, and internet usage
4. Rounding up the call duration (minutes) to the nearest whole minute as per Megaline's recording system

## Data Aggregation and Analysis

Data from each table was grouped by the user_id and the month of usage. Certain metrics were either counted or summed, leading to the calculation of total calls, total call minutes, total messages sent, and total internet usage (in megabytes; rounded) for each user for each month.

We evaluated the average monthly values for each metric for each user and determined the revenue. Subsequently, we plotted the data for the average values of each media from users of both Surf and Ultimate plans. The data was separated based on the plan subscription.

Metrics plotted included:

1. Used call minutes
2. Sent messages
3. Used gigabytes of internet
4. Revenue

We utilized multiple visualization techniques such as bar charts, histograms, and boxplots. The variance, standard deviation, and average monthly values were calculated for each group, allowing us to understand the users' behavior and make sense of the boxplot values.

## Hypotheses Testing

We tested two hypotheses concerning the average monthly revenue for each group:

1. The average monthly revenues from the Surf and Ultimate plans are the same.
2. The average monthly revenue from users in the NY-NJ-PA area is the same as that from users in other regions.

T-tests were applied in both cases, considering whether the variances were equal or not. Eventually, we rejected the hypothesis stating the average revenues from both plans were the same. However, we couldn't reject the hypothesis stating the average revenue in the NY-NJ-PA area was different from other regions. This was consistent with our previous findings.

## Final Takeaway

In general, Ultimate plan subscribers tend to remain within their plan's data limitations. They usually pay their standard $70 service fee and seldom more. Even though this fee is higher than the Surf plan's $20, a significant portion of Surf plan users tend to exceed their monthly limits, resulting in higher overall payments.

A substantial proportion of Surf plan clients overuse their monthly data limits, particularly with internet usage. A quarter of Surf plan clients pay an extra $40 due to internet overuse, resulting in at least a quarter of them paying over $80 to Megaline.

Given the higher number of Surf plan clients and their tendency to exceed data limits, resulting in higher revenue, it would be more profitable to focus marketing efforts on promoting the Surf plan.
