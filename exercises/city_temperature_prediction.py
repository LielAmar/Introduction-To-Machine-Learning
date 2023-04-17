import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=['Date'])
    df.fillna(0)

    # We can drop data that is too odd to be real (samples with -72 degrees)
    df = df[df['Temp'] > -72]

    # Adding the DayOfYear column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # Extracting only Israel's data (and converting Year to be string for discrete color scaling)
    israel_data = df[df['Country'] == 'Israel']
    israel_data = israel_data.astype({'Year': 'str'})

    # First graph/figure
    fig = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                     title="Israel's Temperature as a function of the day of the year")
    fig.update_xaxes(title_text="Day of the Year")
    fig.update_yaxes(title_text="Temperature")
    fig.write_image(f"./ex2_graphs/israel_temp_scatter.png")

    # Second graph/figure
    fig = px.bar(israel_data.groupby(["Month"], as_index=False).agg(std=('Temp', 'std')),
                 x="Month", y="std",
                 title="Israel's Temperature Standard Deviation by Month")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="STD")
    fig.write_image(f"./ex2_graphs/israel_temp_bar.png")

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
