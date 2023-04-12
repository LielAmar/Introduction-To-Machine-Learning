from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    if not X.index.equals(y.index):
        raise IndexError("X and y does not match in their indexes")

    # Remove entries with invalid prices
    if y is not None:
        y = y.loc[y > 0]
        X = X.loc[y.index]

    # Remove redundant features
    # - id: The id of the house
    # - date: Selling date of the house can be ignored
    # - lat & long: Can be ignored because Zipcode is provided
    # - sqft_living15 & sqft_lot15: Data on neighbours can be ignored
    X = X.drop(['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis=1)

    # Remove categorical features
    # - zipcode: The value of zipcode doesn't affect the price directly, so it's categorical
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    # Remove entries that are invalid (NaN/Negative fields)
    X = X.dropna()
    X = X[(X['sqft_living'] > 0) & (X['sqft_lot'] > 0) &
          (X['floors'] > 0) & (X['sqft_above'] > 0)]

    # Add new features for other information we can infer:
    # - is_renovated
    # - is_old
    X['is_renovated'] = np.where((2023 - X['yr_renovated']) < 15, 1, 0)
    X['is_old'] = np.where(((2023 - X['yr_built'] > 30) & (2023 - X['yr_renovated'] > 30)), 1, 0)
    X = X.drop(['yr_renovated'], axis=1)

    if y is not None:
        y = y[X.index]

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:
        f = X[feature]

        # Calculates the Pearson Correlation of every feature and the result series
        # We care about the cov[0][1] (or cov[1][0]) because it holds the relation between the first & second vector
        pc = (np.cov(f, y) / (np.std(f) * np.std(Y)))[0][1]

        # Plotting a scatter graph representing the relation between the feature and the response
        px.scatter(x=f, y=y, trendline="ols",
                   labels=dict(x=f"{feature}", y="Response"),
                   title=f"Pearson Correlation between {feature} and the response is {pc}")\
            .write_image(f"{output_path}/{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    X = df.drop(['price'], axis=1)
    Y = df['price']

    train_X, train_Y, test_X, test_Y = split_train_test(X, Y)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_Y = preprocess_data(train_X, train_Y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_Y, './ex2_graphs')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
