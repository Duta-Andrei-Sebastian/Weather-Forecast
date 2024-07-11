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

## Predictions
To predict how to weather is going to be we are going to train a machine learning algorithm with a linear
regression model.
First, we use the first 1300 entries in the raw data provided to train the algorith. Then test it using the
remaining inputs.

![linear_regression_first_1300_no_scatter](/images/linear_regression_first_1300_no_scatter.png)

In the above graph, the blue line represents the data provided, while the orange line represents the 
prediction made by the algorithm.
The algorithm presents the following parameters:

|               | coef      |
|---------------|-----------|
| year          | 0.133504  |
| month         | -0.043918 |
| temp_min      | 1.259584  |
| precipitation | -0.173899 |
| wind          | -0.272315 |

| method | error              |
|--------|--------------------|
| MSE    | 9.570734696689344  |
| R2     | 0.8334768729365949 |

Secondly, we will try choosing 80% of the given data ata random for training.

![linear_regression_default](/images/linear_regression_default.png)

As in the example above, the blue line represents the data provided, while the orange one represents
the prediction made.
The algorithm presents the following parameters:

|               | coef      |
|---------------|-----------|
| year          | -0.112052 |
| month         | -0.046500 |
| temp_min      | 1.243752  |
| precipitation | -0.170536 |
| wind          | -0.269172 |


| method | error              |
|--------|--------------------|
| MSE    | 11.297832171916616 |
| R2     | 0.7936850590036181 |

Lastly, we will try to get the same results by changing the method. Instead of using a linear regression
method, we will use svr.
We will choose the same data as in the first linear regression training.

![svr](/images/svr.png)

It is obvious that this prediction is nearly identical to the one made with linear regression, but it's slightly
worse. Mathematically this is obvious from the fact that this method's MSE is higher.

| method | error              |
|--------|--------------------|
| MSE    | 9.868544960615035  |
| R2     | 0.8282952125947156 |