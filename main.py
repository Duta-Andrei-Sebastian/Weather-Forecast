import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_info(df: pd.DataFrame):
    print(df.info)


def dataset_info_total(df: pd.DataFrame):
    """ Information regarding the weather
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
    """ Information regarding the weather
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
    """ Histogram of temperature max
    This function makes a histogram of temperature max for each year
    """
    for year in range(2012, 2016):

        plt.hist(df[df['year'] == year]['temp_max'])
        plt.title(year)
        plt.xlabel('Max Temperature')
        plt.ylabel('Count')
        plt.show()


def temp_min_histplot(df: pd.DataFrame):
    """ Histogram of temperature min
    This function makes a histogram of temperature min for each year
    """
    for year in range(2012, 2016):

        plt.hist(df[df['year'] == year]['temp_min'])
        plt.title(year)
        plt.xlabel('Min Temperature')
        plt.ylabel('Count')
        plt.show()

def temp_max_facegrid_lineplot(df: pd.DataFrame):
    """ TODO:
    """

def precipitation_facegrid_scatterplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_countplot(dataframe: pd.DataFrame):
    """ TODO:
    """


def weather_piechart(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_random_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def lr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def svr_predictor_default_split(dataframe: pd.DataFrame):
    """ TODO:
    """


def main():
    df = pd.read_csv('seattle-weather.csv')
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    df.dropna(how='any', axis=0, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.insert(1, "year", df['date'].dt.year, True)
    df.insert(2, "month", df['date'].dt.month, True)
    #dataset_info(df)
    #dataset_info_total(df)
    #dataset_info_by_year(df, 2013)
    #temp_min_histplot(df)
    temp_max_facegrid_lineplot(df)


if __name__ == '__main__':
    main()
