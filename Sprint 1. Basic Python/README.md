# Music Preferences Analysis in Springfield and Shelbyville

## Project Description

In this project, we performed analysis on real Yandex.Music data to understand and compare the music preferences of the cities of Springfield and Shelbyville. The goal was to test a series of hypotheses that could impact business decisions.

## Hypotheses

1. User activity differs depending on the day of the week and from city to city.
2. On Monday mornings, Springfield and Shelbyville residents listen to different genres. This is also true for Friday evenings.
3. Springfield and Shelbyville listeners have different preferences. In Springfield, they prefer pop, while Shelbyville has more rap fans.

## Data Description

The data used for this project is stored in the repository music_project_en.csv.

The dataset includes the following columns:

- 'userID' — user identifier
- 'Track' — track title
- 'artist' — artist's name
- 'genre'
- 'City' — user's city
- 'time' — the exact time the track was played
- 'Day' — day of the week

## Findings

We tested the above hypotheses and came to the following conclusions:

1. **User activity in Springfield and Shelbyville depends on the day of the week**, though the cities vary in different ways. Thus, the first hypothesis is accepted.
2. **Musical preferences do not vary significantly over the course of the week in both Springfield and Shelbyville**. Small differences were observed on Mondays but pop music turned out to be the most popular genre in both cities. Therefore, the second hypothesis cannot be accepted.
3. It seems that the **musical preferences of users from Springfield and Shelbyville are quite similar** with pop music being the top preference in both cities. Thus, the third hypothesis is rejected.

## Conclusion

In conclusion, our hypotheses have been partially proven true. However, while the data suggests some variance in user activity across the week and between the two cities, it shows that the musical preferences of the two cities are largely similar, both favoring pop music over other genres. Please note that conclusions on a city-wide level based on the data from a single source may not always be representative. In real projects, hypothesis testing is more precise and quantitative.

## Future Work

More thorough statistical hypothesis testing would be beneficial in the future to provide more precise and quantitative results. This could lead to more refined and reliable insights about user behavior and preferences. 

> Note: In future projects, we plan to study hypothesis testing in the sprint on statistical data analysis.
