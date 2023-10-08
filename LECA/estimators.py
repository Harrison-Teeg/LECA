import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
## LR
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import preprocessing
from sklearn.utils.validation import check_X_y, check_array
from LECA.prep import to_list
# For type annotations
from typing import List, Tuple, Union, Optional, Callable, Dict

class AlphaGPR(GaussianProcessRegressor):

    """
    Estimator Object which takes a dataframe of the measured objective 
    function and measurement error, and sets the alpha based on the
    measurement error. Otherwise identical to scikit-learn GPR model.
    """

    def __init__(self,
        kernel=None,
        infer_alpha=True,
        min_alpha=False,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        ):
        """
        Constructor for AlphaGPR

        Parameters
        ----------
        \*\*kwargs: kwargs
            Arguments to pass on to GaussianProcessRegressor model (see `docs <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_).
            """
        super(AlphaGPR, self).__init__(kernel = kernel, alpha = alpha,
                optimizer = optimizer, n_restarts_optimizer = n_restarts_optimizer,
                normalize_y = normalize_y, copy_X_train = copy_X_train, random_state = random_state)
        self.min_alpha = min_alpha
        self.infer_alpha = infer_alpha

    def fit(self, X, y) -> GaussianProcessRegressor:
        """
        Identical to fitting GaussianProcessRegressor, except requires a second column in the y DataFrame to define the alpha values for the fit.

        Parameters 
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature vectors.

        y : array-like of shape (n_samples, 2)
            True labels (first column) with measurement error (standard deviations, second column).

        Returns
        -------
            GaussianProcessRegressor
                Fitted (trained) GaussianProcessRegressor object.

        Cite
        ----
        https://scikit-learn.org/stable/developers/develop.html

        """
        # Fit and return
        if isinstance(y, pd.DataFrame):
            stds = np.array(y[y.columns[1]])
            y = y[y.columns[0]]
        else:
            stds = y[:,1]
            y = y[:,0]
        self.alpha = np.power(stds, 2)
        if self.min_alpha:
            self.alpha[self.alpha < self.min_alpha] = self.min_alpha
        self.alpha[np.isnan(self.alpha)] = 1e-10
        super(AlphaGPR, self).fit(X, y)
        return self

class PolynomialRegression(LinearRegression):

    """
    Estimator Object which accepts polynomial feature inputs and selects specified polynomial features to use for fitting.
    """

    def __init__(self,
            polynomials: Optional[Union[int, List[int]]] = None, fit_intercept: bool = True, copy_X: bool = True,
            n_jobs: Optional[int] = None, positive: bool = False
        ):
        """
        Constructor for PolynomialRegression

        Parameters
        ----------
        polynomials: Optional[Union[int, List[int]]]
            int or list of int indices from PolynomialFeatures to use as model features.
            If ``None`` all polynomial features are used.

            Default value ``None``.

        \*\*kwargs: kwargs
            Arguments to pass on to LinearRegression model (see `docs <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_).
            """
        self.polynomials = to_list(polynomials) if polynomials != None else None
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        super(PolynomialRegression, self).__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)

    def fit(self, X, y, sample_weight=None) -> LinearRegression:
        """
        Take polynomial feature input X, select only the declared polynomials from model initialization, and fit to those with a LinearRegression estimator.

        Parameters 
        ----------
        X : array-like of shape (n_samples, n_features)
            Polynomial input feature vectors.

        y : array-like of shape (n_samples, n_objectives)
            True labels.

        sample_weight : array-like of shape (n_samples, n_objectives)
            Sample weights.

        Returns
        -------
            LinearRegression
                Fitted (trained) PolynomialRegression object.

        Cite
        ----
        https://scikit-learn.org/stable/developers/develop.html

        """
        X = self._polynomial_filter(X)
        # Fit and return
        super(PolynomialRegression, self).fit(X, y, sample_weight)
        return self

    def predict(self, X):
        X = self._polynomial_filter(X)
        return super(PolynomialRegression, self).predict(X)

    def _polynomial_filter(self, X):
        if self.polynomials == None or self.polynomials == []: return X
        if isinstance(X, pd.DataFrame): return X.loc[:, self.polynomials]
        return X[:, self.polynomials]
