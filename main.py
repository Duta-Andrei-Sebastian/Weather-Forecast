import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

def dataset_info_total(df: pd.DataFrame):
    """
    Information regarding the weather
    Prints the max and min temperature, the mean max and min temps, the precipitation and its mean value,
    the weather type and how common each one is and the most common one
    """
    print(df['temp_max'])
    print(np.mean(df['temp_max']))
    print(df['temp_min'])
    print(np.mean(df['temp_min']))
    print(df['precipitation'])
    print(np.mean(df['precipitation']))
    print(df['weather'].mode())
    print(df['weather'].value_counts())


def dataset_info_by_year(df: pd.DataFrame, year):
    """
    Information regarding the weather
    Prints the max and min temperature, the mean max and min temps, the precipitation and its mean value,
    the weather type and how common each one is and the most common  for a given year
    """
    print(df[df['year'] == year]['temp_max'])
    print(np.mean(df[df['year'] == year]['temp_max']))
    print(df[df['year'] == year]['temp_min'])
    print(np.mean(df[df['year'] == year]['temp_min']))
    print(df[df['year'] == year]['precipitation'])
    print(np.mean(df[df['year'] == year]['precipitation']))
    print(df[df['year'] == year]['weather'].mode())
    print(df[df['year'] == year]['weather'].value_counts())


def temp_max_histplot(df: pd.DataFrame):
    """
    Histogram of temperature max
    This function makes a histogram of temperature max for each year
    """
    for year in range(2012, 2016):

        plt.hist(df[df['year'] == year]['temp_max'])
        plt.title(year)
        plt.xlabel('Max Temperature')
        plt.ylabel('Count')
        plt.show()


def temp_min_histplot(df: pd.DataFrame):
    """
    Histogram of temperature min
    This function makes a histogram of temperature min for each year
    """
    for year in range(2012, 2016):

        plt.hist(df[df['year'] == year]['temp_min'])
        plt.title(year)
        plt.xlabel('Min Temperature')
        plt.ylabel('Count')
        plt.show()

def temp_max_facegrid_lineplot(df: pd.DataFrame):
    """
    Lineplot of temperature max
    This function plots the temperature max by month for each year
    """
    grid = sns.FacetGrid(df, col='year')
    grid.map(sns.lineplot, 'month', 'temp_max')
    plt.show()


def temp_min_facegrid_lineplot(df: pd.DataFrame):
    """
    Lineplot of temperature min
    This function plots the temperature min by month for each year
    """
    grid = sns.FacetGrid(df, col='year')
    grid.map(sns.lineplot, 'month', 'temp_min')
    plt.show()


def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    """
    Scatterplot of precipitation
    This function plots the precipitation by month for each year
    """
    grid = sns.FacetGrid(dataframe, col='year')
    grid.map(sns.scatterplot, 'month', 'precipitation')
    plt.show()


def weather_countplot(df: pd.DataFrame):
    """
    Countplot of weather
    This function plots the weather to a countplot
    """

    print()
    sns.countplot(x='weather', data=df)
    plt.show()


def weather_piechart(df: pd.DataFrame):
    """
    Piechart of weather
    This function plots the weather
    """
    keys = df['weather'].value_counts().index
    print(keys)
    palette_color = seaborn.color_palette('bright', len(keys))
    plt.pie(df['weather'].value_counts(), labels=keys, colors=palette_color, autopct='%1.1f%%')
    plt.show()

def lr_predictor_random_split(df: pd.DataFrame):
    """ TODO:
    """
    dependent_variable = df['temp_max']
    features = df[['date', 'year', 'month', 'temp_min', 'precipitation', 'wind']]

    x_train, x_test, y_train, y_test = train_test_split(features, dependent_variable, test_size=0.2, random_state=43,shuffle=False)
    reg = LinearRegression().fit(x_train.drop(['date'], axis=1), y_train)
    prediction = reg.predict(x_test.drop(['date'], axis=1))
    print(pd.DataFrame(reg.coef_, features.drop(['date'], axis=1).columns, columns=['coef']))
    MSE = mean_squared_error(y_test, prediction)
    print(MSE)
    R_2 = r2_score(y_test, prediction)
    print(R_2)
    y_test_plot = np.array(y_test)
    x_test_plot = np.array(x_test['date'])
    prediction_plot = np.array(prediction)

    plt.plot(x_test_plot, y_test_plot)
    plt.scatter(x_test_plot, y_test_plot)
    plt.plot(x_test_plot, prediction_plot)
    plt.scatter(x_test_plot, prediction_plot)
    plt.show()



def lr_predictor_default_split(df: pd.DataFrame):
    """ TODO:
    """
    dependent_variable = df['temp_max']
    features = df[['date', 'year', 'month', 'temp_min', 'precipitation', 'wind']]

    x_train, x_test, y_train, y_test = train_test_split(features, dependent_variable, test_size=0.11, random_state=0, shuffle=False)
    reg = LinearRegression().fit(x_train.drop(['date'], axis=1), y_train)
    prediction = reg.predict(x_test.drop(['date'], axis=1))
    print(pd.DataFrame(reg.coef_, features.drop(['date'], axis=1).columns, columns=['coef']))
    MSE = mean_squared_error(y_test, prediction)
    print(MSE)
    R_2 = r2_score(y_test, prediction)
    print(R_2)
    y_test_plot = np.array(y_test)
    x_test_plot = np.array(x_test['date'])
    prediction_plot = np.array(prediction)
    plt.plot(x_test_plot, y_test_plot)
    plt.scatter(x_test_plot, y_test_plot)
    plt.plot(x_test_plot, prediction_plot)
    plt.scatter(x_test_plot, prediction_plot)
    plt.show()


def svr_predictor_default_split(df: pd.DataFrame):
    """ TODO:
    """
    dependent_variable = df['temp_max']
    features = df[['date', 'year', 'month', 'temp_min', 'precipitation', 'wind']]
    x_train, x_test, y_train, y_test = train_test_split(features, dependent_variable, test_size=0.11, random_state=0,
                                                        shuffle=False)
    reg = make_pipeline(StandardScaler(), LinearSVR(tol=1e-5))

    reg.fit(x_train.drop(['date'],axis=1), y_train)

    Y_predicted = reg.predict(x_test.drop(['date'],axis=1))
    y_test_plot = np.array(y_test)
    x_test_plot = np.array(x_test['date'])
    prediction_plot = np.array(Y_predicted)
    plt.plot(x_test_plot, y_test_plot)
    plt.scatter(x_test_plot, y_test_plot)
    plt.plot(x_test_plot, prediction_plot)
    plt.scatter(x_test_plot, prediction_plot)
    plt.show()
    #print(reg.coef_)
    print(f"Mean Squared Error: {mean_squared_error(y_test, Y_predicted)}")
    print(f"r2 score: {r2_score(y_test, Y_predicted)}")


def main():
    df = pd.read_csv('seattle-weather.csv')
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    df.dropna(how='any', axis=0, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.insert(1, "year", df['date'].dt.year, True)
    df.insert(2, "month", df['date'].dt.month, True)
    svr_predictor_default_split(df)


if __name__ == '__main__':
    main()
