from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        error = None

        # Find the best threshold for every feature, with signs 1 and -1
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            thr, thr_err = self._find_threshold(X[:, j], y, sign)

            if error is None or thr_err < error:
                error = thr_err

                self.threshold_ = thr
                self.j_ = j
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        feature = X[:, self.j_]

        return np.where(feature < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        m = len(values)

        # Sort values & labels by the feature value
        index = np.argsort(values)
        values, labels = values[index], labels[index]

        # Loop over every entry in values (each being treated as a possible threshold) and calculate its loss
        loss = np.zeros(shape=(m+1,))

        for i in range(0, m+1):
            # Count the number of elements from 0 to i that we were correct on classifying with -sign,
            # and the number of elements from i+1 to m that we were correct on classifying with sign.
            pre = labels[0:i]
            post = labels[i:m]
            loss[i] = m - (np.size(pre[pre == -sign]) + np.size(post[post == sign]))

        # Find the index with the lowest loss, use values[index] as the threshold and calculate the threshold error
        thr_index = np.argmin(loss)

        pre = labels[0:thr_index]
        post = labels[thr_index:m]

        thr_err = (m - (np.size(pre[pre == -sign]) + np.size(post[post == sign]))) / m

        # If we had the minimal loss at index m, it means that we want to classify all rows with -sign
        if thr_index == m:
            return np.inf, thr_err

        # Otherwise, simply return the value at best_index
        return values[thr_index], thr_err


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        raise NotImplementedError()
