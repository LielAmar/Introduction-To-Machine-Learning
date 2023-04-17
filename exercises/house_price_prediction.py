from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

last_process_cols = []

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

    in_training = (y is not None)

    # Remove redundant features
    # - id: The id of the house
    # - date: Selling date of the house can be ignored
    # - lat & long: Can be ignored because Zipcode is provided
    # - sqft_living15 & sqft_lot15: Data on neighbours can be ignored
    X = X.drop(['id', 'date', 'lat', 'long'], axis=1)
    # , 'sqft_living15', 'sqft_lot15'


    # If we are in training, we are allowed to remove samples that are invalid:
    # Non-Positive price, unbounded values etc.
    if in_training:
        # Remove samples with invalid prices
        y = y.dropna()
        y = y.loc[y.astype(int) > 0]
        X = X.loc[y.index]

        # Remove samples that are invalid (NaN/Negative fields/Values that don't match the field's description)
        X = X.dropna()
        X = X[(X['sqft_living'].astype(float) > 0) & (X['sqft_lot'].astype(float) > 0) &
              (X['sqft_above'].astype(float) > 0) & (X['yr_built'].astype(float) > 0)]

        X = X[(X['bedrooms'].astype(float) >= 0) & (X['bathrooms'].astype(float) >= 0) & (X['floors'].astype(float) >= 0) &
              (X['sqft_basement'].astype(float) >= 0) & (X['yr_renovated'].astype(float) >= 0)]

        X = X[X['waterfront'].astype(int).isin(range(0, 1)) & X['condition'].astype(int).isin(range(1, 6)) &
              X['view'].astype(int).isin(range(0, 5)) & X['grade'].astype(int).isin(range(1, 15))]

        X = X[(X['bedrooms'].astype(float) <= 20) & (X['sqft_lot'].astype(float) <= 1000000)]

        # Keep only prices that match a sample
        y = y.loc[X.index]
    else:
        X = X.fillna(0)

    # Add new features for other information we can infer:
    # - is_renovated
    # - is_old
    X['is_renovated'] = np.where(((2015 - X['yr_renovated'].astype(float)) <= 20), 1, 0)
    # X['is_old'] = np.where(((2015 - X['yr_built'].astype(float) >= 40) &
    #                         (2015 - X['yr_renovated'].astype(float) >= 40)), 1, 0)
    X = X.drop(['yr_renovated'], axis=1)

    global last_process_cols

    # If we're in train, we'd like to save all the columns X has, so we can later on retrieve them on test sets.
    if in_training:
        # Remove categorical features
        # - zipcode: The value of zipcode doesn't affect the price directly, so it's categorical
        X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

        y = y[X.index]

        last_process_cols = X.columns

    # Otherwise, add the missing columns to X, so it can later on be used for testing.
    else:
        for column in last_process_cols:
            if column not in X.columns:
                zc = column.replace("zipcode_", "").replace(".0", "")
                X[column] = np.where(X['zipcode'] == zc, 1, 0)

        for column in X.columns:
            if column not in last_process_cols:
                X = X.drop([column], axis=1)

        # X = X.drop(['zipcode'], axis=1)

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

    y_std = np.std(Y)

    for feature in X.columns:
        if 'zipcode_' in feature: continue

        f = X[feature]

        # Calculates the Pearson Correlation of every feature and the result series
        # We care about the cov[0][1] (or cov[1][0]) because it holds the relation between the first & second vector
        pc = np.around(np.cov(f, y)[0][1] / (np.std(f) * y_std), 3)

        # Plotting a scatter graph representing the relation between the feature and the response
        px.scatter(x=f, y=y, trendline="ols",
                   labels=dict(x=f"{feature}", y="Response"),
                   title=f"Pearson Correlation between {feature} and the response is {pc}") \
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
    # feature_evaluation(train_X, train_Y, './ex2_graphs')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_model = LinearRegression(True)

    # Processing the test data, and keeping only relevant results
    test_X = preprocess_data(test_X, None)[0]
    test_Y = test_Y[test_X.index]

    # Calculating losses for every iteration of every percentage
    losses = np.empty(shape=(91, 10))

    for percentage in range(10, 101):
        for iteration in range(0, 10):
            fit_X = train_X.sample(frac=(percentage / 100))
            fit_Y = train_Y[fit_X.index]

            linear_model.fit(fit_X, fit_Y)

            losses[(percentage - 10), iteration] = linear_model.loss(np.array(test_X), np.array(test_Y))

    # - mean(loss) = mean of every percentage value
    # - std(loss) = sqrt(mean(loss)) = variance(?)
    losses_mean = losses.mean(axis=1)
    losses_var = losses.std(axis=1)

    go.Figure([
            go.Scatter(x=np.arange(10, 101), y=losses_mean, name="Real Mean",
                       mode="markers+lines",
                       marker=dict(color="blue", opacity=1)),
            go.Scatter(x=np.arange(10, 101), y=(losses_mean - 2 * losses_var), name="Error Ribbon",
                       mode="markers+lines", fill='tonexty',
                       marker=dict(color="lightgrey", opacity=.5), line=dict(color="lightgrey")),
            go.Scatter(x=np.arange(10, 101), y=(losses_mean + 2 * losses_var), name="Error Ribbon",
                       mode="markers+lines", fill='tonexty',
                       marker=dict(color="lightgrey", opacity=.5), line=dict(color="lightgrey"))
        ],

        layout=go.Layout(
            title=f"Mean loss as a function of the percentage of data used for fitting the model",
            xaxis={"title": "Percentage of data used for fitting"},
            yaxis={"title": "Mean loss of the test set"})
    ).write_image("./ex2_graphs/percentage_graph.png")
