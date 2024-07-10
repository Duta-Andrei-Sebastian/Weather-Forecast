# Weather Forecast for Seattle

## Preparations

We have been provided with a .cvs file containing data about the weather in 
Seattle during 2012,2013, 2014 and 2015. We must first make sure to eliminate 
duplicate or null rows, as they will interfere with our model.

After doing so, we make sure to convert the date field to the datetime type and
two more columns containing the year and the month.

## First steps
To begin to grasp the way the weather behaves in Seattle we will use the plt.hist()
function to make a histogram of the max and min temperatures for each year:

### 2012
![temp_max_histogram_2012](/images/temp_max_histogram_2012.png)
![temp_min_histogram_2012](/images/temp_min_histogram_2012.png)
### 2013
![temp_max_histogram_2013](/images/temp_max_histogram_2013.png)
![temp_min_histogram_2013](/images/temp_min_histogram_2013.png)
### 2014
![temp_max_histogram_2014](/images/temp_max_histogram_2014.png)
![temp_min_histogram_2014](/images/temp_min_histogram_2014.png)
### 2015
![temp_max_histogram_2015](/images/temp_max_histogram_2015.png)
![temp_min_histogram_2015](/images/temp_min_histogram_2015.png)

Seeing these histograms can help us understand the general way the weather behaves,
but it isn't enough. To gain a good understanding we will use a linegraph created 
using the seaborn module.

![temp_max_lineplot](/images/temp_max_lineplot.png)
![temp_min_lineplot](/images/temp_min_lineplot.png)

To better understand precipitations, we plot them by year with a scatterplot:

![precipitation_scatterplot](/images/precipitation_scatterplot.png)

Because we need to see how the overall weather evolves each day for the 4 years we have
data about, we will use both a piechart and a countplot to help us visualise the data better.

![weather_piechart](/images/weather_piechart.png)
![weather_countplot](/images/weather_countplot.png)

