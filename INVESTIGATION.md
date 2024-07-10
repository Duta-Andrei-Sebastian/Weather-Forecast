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

![max_temperature_histogram_2012](/images/max_temperature_histogram_2012.png)
![min_temperature_histogram_2012](/images/min_temperature_histogram_2012.png)
![max_temperature_histogram_2013](/images/max_temperature_histogram_2013.png)
![min_temperature_histogram_2013](/images/min_temperature_histogram_2013.png)
![max_temperature_histogram_2014](/images/max_temperature_histogram_2014.png)
![min_temperature_histogram_2014](/images/min_temperature_histogram_2014.png)
![max_temperature_histogram_2015](/images/max_temperature_histogram_2015.png)
![min_temperature_histogram_2015](/images/min_temperature_histogram_2015.png)

Seeing these histograms can help us understand the general way the weather behaves,
but it isn't enough. To gain a good understanding we will use a linegraph created 
using the seaborn module    



