# Summary

I tried to explain the changes in flight delays in the USA between 1987 and 2008 by weekdays. The first animated graph shows the average total (i.e. arrival + departure) delay in days for each year. The second graph shows the changes in departure, arrival or total delay throughout the years for each day, days are represented by colours.

# Design

## Data Preparation
The original data is more than 10GB and has about 120 million rows in 22 files for each year with 29 attributes. Initially, I combined the files by keeping the attributes I need for my visualization by using [this](https://github.com/ddaskan/Data-Analyst-Nanodegree/blob/master/P6-Data_Visualization/data/combiner3.py) code. Since the data is still huge, I, then, aggregated it by grouping by 'Year' and 'Day' to obtain the mean values of 'ArrDelay' and 'DepDelay' (i.e. delays on arrival and departure respectively) and I added a new feature called 'TotalDelay' to obtain the total delay per flight by using [this](https://github.com/ddaskan/Data-Analyst-Nanodegree/blob/master/P6-Data_Visualization/data/Agg.ipynb) code.

# Feedback

1. 
2. 
3. 

# Resources
* [dimple.js](http://dimplejs.org/)
* [dimple Wiki](https://github.com/PMSI-AlignAlytics/dimple/wiki)
* [Data](http://stat-computing.org/dataexpo/2009/the-data.html)
