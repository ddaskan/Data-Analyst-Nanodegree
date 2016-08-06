# Summary
I tried to explain the changes in flight delays in the USA between 1987 and 2008 by weekdays. The first animated graph shows the average total (i.e. arrival + departure) delay in days for each year. The second graph shows the changes in departure, arrival or total delay throughout the years for each day, days are represented by colours. It looks Friday leads followed by Thursday for almost every year.

# Design
### Data Preparation
The original data is more than 10GB and has about 120 million rows in 22 files for each year with 29 attributes. Initially, I combined the files by keeping the attributes I need for my visualization by using [this](https://github.com/ddaskan/Data-Analyst-Nanodegree/blob/master/P6-Data_Visualization/data/combiner3.py) code. Since the data is still huge, I, then, aggregated it by grouping by 'Year' and 'Day' to obtain the mean values of 'ArrDelay' and 'DepDelay' (i.e. delays on arrival and departure) and I added a new feature called 'TotalDelay' to obtain the total delay per flight by using [this](https://github.com/ddaskan/Data-Analyst-Nanodegree/blob/master/P6-Data_Visualization/data/Agg.ipynb) code.

### Visualization
I chose the bubbles and line graph to visualize and present my findings for delay changes by year and day respectively. I created four charts in total initially.

1. An animated bubble graph with total delays in y axis and days in x axis. It changes for every year.
2. A line graph showing only delays on departure with delays in y axis and years in x axis. Days are denoted in different coloured lines.
3. A line graph showing only delays on arrival with delays in y axis and years in x axis. Days are denoted in different coloured lines.
4. A line graph showing only total delays with delays in y axis and years in x axis. Days are denoted in different coloured lines.

The initial design can be seen [here](http://bl.ocks.org/ddaskan/raw/d0a55c4f14f19c421000127746c0b41e/).  

After received feedback, I did some changes on my design.

So the final design can be seen [here]().

# Feedback
1. There are too much graphs, you can combine last 3. And you can make the year legend better by centering years in blue areas in the first graph.
2. I don't understand y axis titles and I guess you can use at least 1 decimal for values.
3. You need to change dots to lines for first graph because it's time series, rest look okay.

# Resources
* [dimple.js](http://dimplejs.org/)
* [dimple Wiki](https://github.com/PMSI-AlignAlytics/dimple/wiki)
* [Data](http://stat-computing.org/dataexpo/2009/the-data.html)
* [stackoverflow](http://stackoverflow.com/)