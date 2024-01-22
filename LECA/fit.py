import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
## NN
from sklearn.neural_network import MLPRegressor
## RF
from sklearn.ensemble import RandomForestRegressor
## LR
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from LECA.estimators import PolynomialRegression, AlphaGPR
from sklearn import preprocessing
## GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF, Matern, WhiteKernel, DotProduct, ExpSineSquared,RationalQuadratic
## Metrics
from scipy import stats
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline # use pipelines in cross_validation to avoid test/train leakage from data-scaling
#Recursive feature elimination using cross-validation
from sklearn.feature_selection import RFECV
from LECA.prep import to_list
## Bayesian Optimization
import GPy, GPyOpt
## MAPIE uncertainty estimation
from mapie.regression.regression import MapieRegressor
from mapie.subsample import Subsample
## Uncertainty propagation
from uncertainties import unumpy, ufloat
import random
# For type annotations
from typing import List, Tuple, Union, Optional, Callable, Dict
from sklearn.base import BaseEstimator
# Optimization
import scipy.stats as st
import scipy.optimize as opt
from itertools import product
# Ranked batch mode active learning
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler

def score(predicted, true_value) -> Dict[str, float]:
    """
    A shortcut to call ``sklearn.metrics`` (see `scikit-learn docs <https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics>`_) to return a dictionary of the r2, MAE, MSE, and RMSE scores for a given set of predictions `predicted` compared to the true values `true_value`.

    Parameters
    ----------
        predicted : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.

        true_value : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.

    Returns
    -------
        Dict[str, float]
            Dictionary in the form:

            =========== ================
            'r2'        float
            'MAE'       float
            'MSE'       float
            'RMSE'      float
            =========== ================
    """

    return {'r2': r2_score(true_value,predicted),
            'MAE': mean_absolute_error(true_value,predicted),
            'MSE': mean_squared_error(true_value,predicted),
            'RMSE': np.sqrt(mean_squared_error(true_value,predicted))}



class WorkFlow:
    """
    Fundamental object for training/optimizing LECA regression models, tracking their performance via cross-validation and bootstrapping datasets for error-estimation.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing full feature set and objective functions.

    features: Union[str, List[str]]
        str or List[str] enumerating the features to be used by the regression models. Must match column names in the `data` DataFrame.

    objective_list: Union[str, List[str]]
        str or List[str] enumerating the objective functions (i.e. target values) for the regression task. Must match column names in the `data` DataFrame.

    random_state: Optional[int]
        Sets the ``random_state`` parameter for any stochastic process used in the regression models for reproducibility.

        Default value ``None``.

    polynomial_degree: int
        Sets the maximum degree for the polynomial features generated for linear regression (see `sklearn.preprocessing.PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_).

        Default value ``3``.

    validation_holdout: Union[int, float]
        Select number of datapoints to hold out as a validation set (unseen by regression models). An ``int`` holdout declares an explicit number of datapoints to exclude, while a ``float`` is then the fraction of the total dataset to reserve for validation.

        Default value ``0``.

    composition_features: Optional[Union[str, List[str]]]
        str or List[str] defining which features characterize a unique composition. This parameter is for test/train/validation split grouping purposes. For example, an electrolyte would have salt / solvent / additive concentrations as composition features, but temperature would be exempt. By defining the composition_features as the component concentrations we can then group each unique composition and split test/train/validation sets to avoid overfitting by including an identical electrolyte composition in the training set at only a slightly different temperature than in the test/validation sets.

        Default value ``None``.

    Attributes
    ----------
    supported_models: List[str]
        List of supported regression model names (i.e. `regr_name` in :meth:`.add_regr`).

        =========================== ==================
        Type                        Name
        Neural Network              "nn"
        Random Forest               "rf"
        Linear Reg.                 "poly_lr"
        Lasso Lin. Reg.             "lasso_lr"
        Ridge Lin. Reg.             "ridge_lr"
        GPR - Iso RBF               "gpr_RBF_iso"
        GPR - Aniso RBF             "gpr_RBF_aniso"
        GPR - Iso Matern            "gpr_Matern_iso"
        GPR - Aniso Matern          "gpr_Matern_aniso"
        GPR - Rational Quadratic    "gpr_RQ_iso"
        GPR - Custom Kernel         "gpr_custom"
        =========================== ==================

    supported_metrics: List[str]
        List of supported scoring metrics for regression models. Currently supported metrics are: "r2", "MAE", "MSE", and "time".

    features: List[str]
        List[str] of the features used by the regression models.

    objective_list: List[str]
        List[str] of the objective functions for this WorkFlow instance.

    X: pd.DataFrame
        DataFrame sliced from the DataFrame provided during WorkFlow initialization. Columns are the declared features, rescaled to have a normal distribution with a mean of 0, with each row corresponding to a measured objective function value. (uses: `scikit-learn StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_)

    poly_X: pd.DataFrame
        DataFrame of mixed polynomial features built from the ``X`` DataFrame. All polynomial combinations up to a max degree of ``polynomial_degree`` are generated. (uses: `scikit-learn PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_)

    X_unscaled: pd.DataFrame
        DataFrame of the raw input features, analogous to the ``X`` attribute but unscaled.

    X_validate: pd.DataFrame
        DataFrame, analog to ``X_unscaled`` of the raw input features set aside for the validation set (unseen by ML models and excluded from ``X``, ``X_unscaled`` and ``poly_X``)
        
    y, y_validate, std, std_validate: pd.DataFrame
        Grouped for brevity, analogous to the ``X`` DataFrame set. Each attribute is a DataFrame storing either the objective function values (rows corresponding to the rows of the ``X...`` DataFrames) and their standard deviations (if measurements were repeated and combined to have columns in the initialization DataFrame with the name `\<objective function name\>_std`) ``..._validate`` is then the group of values split from the main set to be withheld for validation purposes. ``y`` and ``std`` support multiple objective functions.

    _random_state: Optional[int]
        Optional value declared at WorkFlow initialization and passed to any functions/methods called by the WorkFlow which accept a 'random_state'.

    _n_jobs: Optional[int]
        Optional value declared at WorkFlow initialization and passed to any functions/methods called by the WorkFlow which accept a parallelization parameter to use `n` cores. Note ``-1`` will use all available cores. ``None`` uses only one core.

    results: Dict
        Dictionary with a key for each objective function declared at initialization. This dictionary stores the ML models trained for the corresponding objective function and their performance scores.
    """

    supported_models = ['nn',
                        'rf',
                        'poly_lr',
                        'lasso_lr',
                        'ridge_lr',
                        'gpr_RBF_iso',
                        'gpr_RBF_aniso',
                        'gpr_Matern_iso',
                        'gpr_Matern_aniso',
                        'gpr_RQ_iso',
                        'gpr_custom']

    supported_metrics = ['r2', 'MAE', 'MSE', 'time']

    def __init__(
        self, data: pd.DataFrame, features: Union[str, List[str]],
        objective_list: Union[str, List[str]], random_state: Optional[int] = None,
        n_jobs: Optional[int] = -1, polynomial_degree: int = 3,
        validation_holdout: Union[int, float] = 0,
        composition_features: Optional[Union[str, List[str]]] = None
    ) -> None:

        objective_list = to_list(objective_list)
        X = data[features]
        y = data[objective_list]
        std = data[[obj + "_std" for obj in objective_list if obj + "_std" in data.columns]]

        # Create a dataframe of data index -> group (where group represents a unique ID for a unique composition)
        if composition_features != None: groups = X.groupby(composition_features).ngroup()
        
        ## Extract validation set
        if validation_holdout > 0:
            if composition_features != None: # If we want to train/validate split considering composition groups:
                gss = GroupShuffleSplit(n_splits=1, test_size=validation_holdout, random_state=random_state)
                train_index, validate_index = next(gss.split(X, groups=groups)) # 1 element generator
                #Note, change X_val before X of course as X is source
                X_validate = X.iloc[validate_index]
                X = X.iloc[train_index]
                y_validate = y.iloc[validate_index]
                y = y.iloc[train_index]
                std_validate = std.iloc[validate_index]
                std = std.iloc[train_index]
                self.groups_validate = groups.iloc[validate_index].reset_index(drop=True)
                groups = groups.iloc[train_index]
                # We still need to shuffle our training data
                X,y,std,groups = shuffle(X,y,std,groups, random_state=random_state)
                self.groups = groups.reset_index(drop=True)
            else:
                X, X_validate, y, y_validate, std, std_validate = train_test_split(X,y,std, test_size=validation_holdout, random_state=random_state)
        # If validation_holdout == 0: We still need to shuffle our data
        else:
            if composition_features != None: # If we want to train/validate split considering composition groups:
                X,y,std,groups = shuffle(X,y,std,groups, random_state=random_state)
                self.groups = groups.reset_index(drop=True)
            else:
                X,y,std = shuffle(X,y,std, random_state=random_state)
        

        #Generate polynomials
        self.poly_transformer = preprocessing.PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        poly_X = pd.DataFrame(self.poly_transformer.fit_transform(X))

        ##Scale (on training set to avoid test/val info leaking into models)
        self.scaler = preprocessing.StandardScaler().fit(X) # Scale 
        self.poly_scaler = preprocessing.StandardScaler().fit(poly_X)
        self.X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        self.poly_X = pd.DataFrame(self.poly_scaler.transform(poly_X))
     
        # Save rest of attributes
        self.X_unscaled = X
        self.poly_unscaled = poly_X
        self.y = y
        self.std = std
        if validation_holdout > 0:
            self.X_validate_unscaled = X_validate
            self.poly_validate_unscaled = self.poly_transformer.transform(X_validate)
            self.X_validate = pd.DataFrame(self.scaler.transform(X_validate), columns=X.columns)
            self.poly_validate = pd.DataFrame(self.poly_scaler.transform(self.poly_transformer.transform(X_validate)))
            self.y_validate = y_validate
            self.std_validate = std_validate
        self._random_state = random_state
        self._n_jobs = n_jobs
        self.results = {objective_function : {} for objective_function in y}
        self.features = features
        self.objective_list = objective_list
        self.composition_features = composition_features
        self.data = data
    
        #Initialize GPR kernels
        aniso_length_scale = np.ones(X.shape[1]) # l = 1 for each dimension
        self._kernels = {
            "RBF_iso" : ConstantKernel()*RBF() + ConstantKernel(),
            "RQ_iso" : ConstantKernel()*RationalQuadratic() + ConstantKernel(),
            "Matern_iso" : ConstantKernel()*Matern() + ConstantKernel(),
            "RBF_aniso" : ConstantKernel()*RBF(length_scale=aniso_length_scale) + ConstantKernel(),
            "Matern_aniso" : ConstantKernel()*Matern(length_scale=aniso_length_scale) + ConstantKernel(),
            }

    def reinit_data_sets(self, random_state: Optional[int] = None,
        validation_holdout: Union[int, float] = 0, 
    ) -> None:
        """
        Reset the WorkFlow data splitting / shuffling with (optionally) modified random_state
        and validation_holdout values.
    
        Parameters
        ----------
        
        random_state: Optional[int]
            Sets the ``random_state`` parameter for any stochastic process used in the regression models for reproducibility.
            
        validation_holdhout: Optional[Union[int, float]]
            Sets the split for train and test set. If an int is given, a fixed number of data points is stored in the validation_holdout set. 
            If a float is given, a percentage of data points is stored in the validation_holdout set.
        
        Returns
        -------
            ``None``
    
        """
        objective_list = self.objective_list
        features = self.features
        composition_features = self.composition_features
        data =  self.data.copy()
        
        self._random_state = random_state
        
        X = self.data[features]
        y = self.data[objective_list]
        std = data[[obj + "_std" for obj in objective_list if obj + "_std" in data.columns]]

        # Create a dataframe of data index -> group (where group represents a unique ID for a unique composition)
        if composition_features != None: groups = X.groupby(composition_features).ngroup()
        
        ## Extract validation set
        if validation_holdout > 0:
            if composition_features != None: # If we want to train/validate split considering composition groups:
                gss = GroupShuffleSplit(n_splits=1, test_size=validation_holdout, random_state=random_state)
                train_index, validate_index = next(gss.split(X, groups=groups)) # 1 element generator
                #Note, change X_val before X of course as X is source
                X_validate = X.iloc[validate_index]
                X = X.iloc[train_index]
                y_validate = y.iloc[validate_index]
                y = y.iloc[train_index]
                std_validate = std.iloc[validate_index]
                std = std.iloc[train_index]
                self.groups_validate = groups.iloc[validate_index].reset_index(drop=True)
                groups = groups.iloc[train_index]
                # We still need to shuffle our training data
                X,y,std,groups = shuffle(X,y,std,groups, random_state=random_state)
                self.groups = groups.reset_index(drop=True)
            else:
                X, X_validate, y, y_validate, std, std_validate = train_test_split(X,y,std, test_size=validation_holdout, random_state=random_state)
        # If validation_holdout == 0: We still need to shuffle our data
        else:
            if composition_features != None: # If we want to train/validate split considering composition groups:
                X,y,std,groups = shuffle(X,y,std,groups, random_state=random_state)
                self.groups = groups.reset_index(drop=True)
            else:
                X,y,std = shuffle(X,y,std, random_state=random_state)
        

        #Generate polynomials
        poly_X = pd.DataFrame(self.poly_transformer.fit_transform(X))

        ##Scale (on training set to avoid test/val info leaking into models)
        self.X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        self.poly_X = pd.DataFrame(self.poly_scaler.transform(poly_X))
     
        # Save rest of attributes
        self.X_unscaled = X
        self.poly_unscaled = poly_X
        self.y = y
        self.std = std
        if validation_holdout > 0:
            self.X_validate_unscaled = X_validate
            self.poly_validate_unscaled = self.poly_transformer.transform(X_validate)
            self.X_validate = pd.DataFrame(self.scaler.transform(X_validate), columns=X.columns)
            self.poly_validate = pd.DataFrame(self.poly_scaler.transform(self.poly_transformer.transform(X_validate)))
            self.y_validate = y_validate
            self.std_validate = std_validate
            
    def retrain(self):
        """
        Retrains all models of the WorkFlow. Cross validation scores and uncertainties from previous training are removed. 
        
        Returns
        -------
            ``None``
        
        """
    
        for obj in self.objective_list:
            for regr_name in self.results[obj].keys():
                model = self.results[obj][regr_name]['model']
                if hasattr(model, 'random_state'):
                    model.random_state = self._random_state
                if isinstance(model, PolynomialRegression): 
                    model.fit(self.poly_X, self.y[obj])
                elif isinstance(model, AlphaGPR): 
                    alpha_y = pd.concat([self.y[obj], self.std[obj+'_std']],axis=1) # alphaGPR takes y as col1: objective, col2: objective_err
                    model.fit(self.X, alpha_y)
                else:
                    model.fit(self.X, self.y[obj])
                self.results[obj][regr_name]['model'] = model
                self.results[obj][regr_name]['metrics'] = False # False until scored with cross_validate
                if not isinstance(model, GaussianProcessRegressor):
                    self.results[obj][regr_name]['uncertainty'] = False # False until estimated with estimate_uncertainty
            
    def add_regr(
        self, fit_name: str, regr_name: str, objective_funcs: Optional[Union[str, List[str]]] = None, **hyperparameters
    ) -> None:
        """
        Adds a regression model to the WorkFlow. See \*\*kwargs parameter description below for further info on LECA specific hyperparameters `infer_alpha, polynomials` and `degree`. 

        Parameters
        ----------
        fit_name: str
            Unique identifying string for regression model. If the `fit_name` already exists for the enumerated `objective_funcs` it/they will be overwritten by calling this function.

        regr_name: str
            Regression model type. Must be a supported model (see ``LECA.WorkFlow.supported_models``).

        objective_funcs: Optional[Union[str, List[str]]] 
            Objective function(s) regression model should predict. The string or list of strings given must match the name(s) of an enumerated objective function during the creation of the WorkFlow object. If ``None`` a regression model for each objective function loaded into the workflow will be created.

        \*\*hyperparameters: kwargs
            Kwargs hyperparameters to be passed on when creating the regression model objects. Aside from the LECA specific model hyperparameters listed below, hyperparameters are passed on to the scikit-learn model object and the user is directed to the relevant `scikit-learn docs <https://scikit-learn.org/stable/supervised_learning.html>`_.

            `infer_alpha` : ``Boolean`` hyperparameter for GPR models
                Hyperparameter for GPR type models. If ``False``, the models are constructed directly as scikit-learn objects with the specified kernel (+ WhiteKernel). If passed as ``True``, LECA uses the extended GPR model clase :class:`AlphaGPR`. This is essentially identical to the standard scikit-learn GPR model but it takes two objective functions when training. y, and y_std. y_std is then stored as the alpha array (representing training data biases) and the model is trained on the objective values stored in the y array. **AlphaGPR models are trained with no white noise in the kernel.**

            `polynomials` : ``List[int]`` hyperparameter for polynomial linear regression models (`poly_lr`)
                LECA has its own modified implementation for polynomial regression (see ``LECA.estimators.PolynomialRegression``) which allows the user to specify which polynomials (as a list of indices corresponding to polynomial feature in `WorkFlow.poly_X`) are used.

            `degree` : ``int`` hyperparameter for polynomial linear regression models (`poly_lr`)
                Accepts an integer value which will automatically include all polynomials less than or equal to the given degree (i.e. autofills the `polynomials` indices).

        Returns
        -------
            ``None``

        """
         ## Default to all objective functions if none are provided
        if objective_funcs == None: objective_funcs = list(self.y.columns)

        # Some useful local variables
        objective_list = list(self.y.columns)
        X = self.X
        y = self.y
        std = self.std
        poly_X = self.poly_X
        
        if not regr_name in self.supported_models:
            print(" \'{}\' invalid regression model, try one of: {}".format(regr_name, self.supported_models))
            return
        if regr_name == 'gpr_custom' and not 'kernel' in hyperparameters.keys():
            print(" \'{}\' requires a kernel in the form of kernel=<kernel object>".format(regr_name))
            return
        
        ## Train the given regression model each objective in the list of objective funcs given
        for obj in to_list(objective_funcs):
            if not obj in objective_list:
                print(" \'{}\' invalid target to fit, try one of: {}".format(obj, objective_list))
                continue
        
            #Initialize data storage OR, if already existing, clear:
            self.results[obj][fit_name] = {}
            store = self.results[obj][fit_name]
            store['hyperparameters'] = hyperparameters
            store['metrics'] = False# False until scored with cross_validate
            store['uncertainty'] = False# False until estimated with estimate_uncertainty
            
            ## A lot of this is repetition -- but some is not...
            if regr_name == 'nn':
                if not 'max_iter' in hyperparameters.keys(): hyperparameters['max_iter'] = 1000
                store['model'] = MLPRegressor(**hyperparameters, random_state=self._random_state).fit(X, y[obj])
            elif regr_name == 'rf':
                store['model'] = RandomForestRegressor(**hyperparameters, random_state=self._random_state, n_jobs=self._n_jobs).fit(X, y[obj])
            elif regr_name.startswith('gpr_'):
                # If we've passed a custom-defined kernel, use that
                if 'kernel' in hyperparameters.keys():
                    if 'infer_alpha' in hyperparameters.keys():
                        alpha_y = pd.concat([y[obj], std[obj+'_std']],axis=1) # alphaGPR takes y as col1: objective, col2: objective_err
                        store['model'] = AlphaGPR(**hyperparameters, random_state=self._random_state).fit(X, alpha_y)
                    else:
                        store['model'] = GaussianProcessRegressor(**hyperparameters, random_state=self._random_state).fit(X, y[obj])
                # Otherwise, auto-optimize kernel:
                else:
                    #TODO: Don't forget to decide whether we -always- include a whitekernel or only for non-alpha-gpr!!!
                    kernel = self._kernels[regr_name[4:]] + WhiteKernel(noise_level_bounds=(1e-15,1e5))

                    ## First check if we want to use the alpha_inference technique for lower prediction errors (but higher risk of overfitting)
                    if not 'infer_alpha' in hyperparameters.keys() or hyperparameters['infer_alpha'] == False:
                        if 'infer_alpha' in hyperparameters.keys(): hyperparameters.pop('infer_alpha') # leaving infer_alpha out of the hyperparameters is the same as it being false
                        ## Then regular fit with WhiteNoise
                        #kernel = kernel + WhiteKernel(noise_level_bounds=(1e-15,1e5))
                        store['model'] = GaussianProcessRegressor(kernel=kernel, **hyperparameters, random_state=self._random_state).fit(X, y[obj])
                    else:
                        # Note infer_alpha doesn't use WhiteKernel by default
                        alpha_y = pd.concat([y[obj], std[obj+'_std']],axis=1) # alphaGPR takes y as col1: objective, col2: objective_err
                        store['model'] = AlphaGPR(kernel=kernel, **hyperparameters, random_state=self._random_state).fit(X, alpha_y)

                        # Old, depreciated infer_alpha behavior
                        #custom_noise = WhiteKernel()
                        ## If we have deviations calculated from measurement data, set this as the minimum noise for fit
                        ### increase restarts to ensure optimal kernel hyperparams
                        #if not 'n_restarts_optimizer' in local_params.keys(): local_params['n_restarts_optimizer'] = 0
                        # If we have std data from our objective function, set the measurement error std^2 as lower noise bound
                        #if obj + '_std' in self.std.columns:
                        #    lower_bound = max(np.power(self.std[obj+'_std'],2).mean(), 1e-8) ## Minimum noise level
                        #    custom_noise = WhiteKernel(noise_level_bounds=(lower_bound,lower_bound*1e5))

                        ## Fit model with custom_noise
                        #gpr_model = GaussianProcessRegressor(kernel = kernel + custom_noise, **local_params, random_state=self._random_state).fit(X, y[obj])

                        ## Extract noise from whitekernel and substitute into alpha, retrain model
                        #optimized_noise = [kernel for kernel in gpr_model.kernel_.get_params().values() if isinstance(kernel, WhiteKernel)][0].get_params()['noise_level']
                        #local_params['alpha'] = optimized_noise
                        #print("Optimized noise level for {} with model {} estimated as {}".format(obj, fit_name, optimized_noise))
                        #local_params['n_restarts_optimizer'] = 5
                        #local_params['kernel'] = kernel
                        #gpr_model = GaussianProcessRegressor(**local_params, random_state=self._random_state).fit(X, y[obj])

                        ## Fix kernel values from previous fit by setting optimizer=None
                        #local_params['kernel'] = gpr_model.kernel_
                        #local_params['optimizer'] = None
                        #local_params['n_restarts_optimizer'] = 0
                        #gpr_model = GaussianProcessRegressor(**local_params, random_state=self._random_state).fit(X, y[obj])
                        #store['model'] = gpr_model

                store['uncertainty'] = True# gpr has inbuilt uncertainty
            elif regr_name == 'poly_lr' or regr_name == 'lasso_lr' or regr_name == 'ridge_lr':
                poly_transformer = self.poly_transformer
                ## List of all polynomials used (by column # of poly_values DataFrame), if empty, use all
                polynomials_indices = []
                # If a simple degree is given, add all polynomials <= degree to polynomials_indices list
                if 'degree' in hyperparameters.keys():
                    degree = hyperparameters.pop('degree')
                    generated_degree = poly_transformer.get_params()['degree']
                    if degree > generated_degree: raise ValueError("Polynomial features were only generated to the degree: " + str(generated_degree))
                    ## Convert degree -> polynomials_indices list
                    power_list = poly_transformer.powers_
                    for i in range(len(power_list)):
                        if np.sum(power_list[i]) <= degree: polynomials_indices.append(i)
                        else: break

                ## If we explicitly declare which polynomials we want to use, add those as well
                if 'polynomials' in hyperparameters.keys():
                    polynomials = to_list(hyperparameters.pop('polynomials'))
                    for polynomial_index in polynomials:
                        if not polynomial_index in polynomials_indices: polynomials_indices.append(polynomial_index)
                ## Store the polynomials to use as a hyperparameter
                if len(polynomials_indices) > 0: store['hyperparameters']['polynomials'] = polynomials_indices
                else: store['hyperparameters']['polynomials'] = list(range(len(self.poly_transformer.powers_))) # if we pass no declaration of polynomials, use all
                store['model'] = PolynomialRegression(**hyperparameters).fit(poly_X, y[obj])
                

    def remove_regr(
        self, fit_name: str, objective_funcs: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Removes a regression model from the WorkFlow object.

        Parameters
        ----------
        fit_name: str
            Unique model identifier name.

        objective_funcs: Optional[Union[str, List[str]]]
            String or list of strings enumerating the objective functions for which the models should be deleted.
            If ``None``, all objective functions will have the associated named model removed.

            Default value ``None``.

        Returns
        -------
            ``None``
        """
        ## Default to all objective functions if none are provided
        if objective_funcs == None: objective_funcs = list(self.y.columns)

        # Some useful local variables
        objective_list = list(self.y.columns)
        for obj in to_list(objective_funcs):
            if not obj in objective_list:
                print(" \'{}\' invalid target to fit, try one of: {}".format(obj, objective_list))
                continue
        
            #Remove value
            self.results[obj].pop(fit_name, None)#Remove value, return None if it doesn't exist
        
    def estimate_uncertainty(self,
        regressions: Optional[str] = None, objective_funcs: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        Method to enable prediction uncertainty estimation for regression models.
        For GPR models this method will do nothing.
        For PolynomialRegression models 200 bootstrapped models are trained.
        For any other, 30 models are trained.

        This method implements `MapieRegressor <https://mapie.readthedocs.io/en/latest/generated/mapie.regression.MapieRegressor.html#mapie.regression.MapieRegressor>`_ to generate bootstrapped models and enable uncertainty estimation using Jackknife+-After-bootstrap for non-GPR models. For further details see: `Mapie jackknife+-AB <https://mapie.readthedocs.io/en/latest/theoretical_description_regression.html#the-jackknife-after-bootstrap-method>`_

        Parameters
        ----------
        fit_name: str
            Regression model unique identifying name.

        objective_funcs: Optional[Union[str, List[str]]]
            String or list of strings enumerating the objective functions for which the bootstrapped models should be trained.
            If ``None``, the named bootstrapped models on all objective functions will be trained.

            Default value ``None``.

        Returns
        -------
            ``None``
        """

        ## Default to all objective functions if none are provided
        if objective_funcs == None: objective_funcs = list(self.y.columns)

        # Some useful local variables
        objective_list = list(self.y.columns)
        X = self.X
        y = self.y
        poly_X = self.poly_X
        
        ## Train the given regression model each objective in the list of objective funcs given
        for obj in to_list(objective_funcs):
            if not obj in objective_list:
                print(" \'{}\' invalid target to score, try one of: {}".format(obj, objective_list))
                continue
            for regr_name, store in self.results[obj].items():
                if not (regressions == None or regr_name in to_list(regressions)): continue
                if not store['uncertainty']:
                    regr = store['model']
                    print(regr)
                    mapie_X = X
                    resamplings = 30
                    if isinstance(regr, PolynomialRegression):
                        mapie_X = poly_X
                        resamplings = 200
                    mapie = MapieRegressor(regr, **{"method":"plus", 'cv':Subsample(n_resamplings=resamplings, random_state=self._random_state)}, n_jobs=self._n_jobs)
                    mapie.fit(mapie_X, y[obj])
                    store['uncertainty'] = mapie

    def cross_validate(self,
            cv: int = 5, objective_funcs: Optional[Union[str, List[str]]] = None, verbose: bool = True
        ) -> None:
        """
        Method to score regression model performance with k-fold cross validation.

        If `composition_features` are defined for the WorkFlow, the *grouped* inputs
        are split into CV-folds, rather than individual data, i.e. each group of
        data with identical `composition_features` are assigned together to either
        the training or test set for each fold (to prevent data leakage).

        Parameters
        ----------
        cv : int
            Number of cross validation folds.

        objective_funcs: Optional[Union[str, List[str]]]
            Name or list of names of objective functions to score. All regression models initiated for these objective functions will have k-fold cross validated scores recorded. If ``None`` then all objective functions defined for the WorkFlow will be scored.

            Default value ``None``.

        verbose: bool
            Toggles whether to output scores (instead of storing them in the workflow metrics database).

            Default value ``True``.

        Returns
        -------
            ``None``
        """

        ## Default to all objective functions if none are provided
        if objective_funcs == None: objective_funcs = list(self.y.columns)

        # Some useful local variables
        objective_list = list(self.y.columns)
        X = self.X_unscaled                     # unscaled since we now use pipeline to rescale within the cv fold
        y = self.y
        std = self.std
        poly_X = self.poly_unscaled             # unscaled since we now use pipeline to rescale within the cv fold
        
        ## Train the given regression model each objective in the list of objective funcs given
        for obj in to_list(objective_funcs):
            if not obj in objective_list:
                print(" \'{}\' invalid target to score, try one of: {}".format(obj, objective_list))
                continue
            for regr_name, store in self.results[obj].items():
                if not store['metrics']:
                    model = store['model']
                    regr = Pipeline([('scaler', preprocessing.StandardScaler()), ('model', model)])
                    cv_X = X
                    loc_y = y[obj]
                    return_estimator = False
                    ## If we have a poly LR model -- Store the trained model coefficients
                    if isinstance(model, PolynomialRegression):
                        cv_X = poly_X

                    if verbose: print(regr)
                    ## Return the estimator for later capturing deviations between different CV trained models
                    if hasattr(self, 'groups'):
                        n_samples = len(self.groups.unique())
                        if cv > n_samples:
                            print('Reducing CV-folds to ' + str(n_samples) + ', since dataset only has that many unique samples')
                            cv = n_samples
                        if isinstance(model, AlphaGPR):
                            loc_y = pd.concat([y[obj], std[obj+'_std']], axis=1)

                            ## Custom scorers to strip away the deviations we pass through along with y (so the pipeline doesn't rescale them during CV_scoring)
                            def r2(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return r2_score(y[:,0], y_pred)
                            def neg_square(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return - mean_squared_error(y[:,0], y_pred)
                            def neg_abs(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return - mean_absolute_error(y[:,0], y_pred)

                            store['metrics'] = cross_validate(regr, cv_X, loc_y, cv=GroupKFold(cv),
                                    groups=self.groups, return_train_score=True, scoring={'neg_mean_absolute_error':neg_abs, 'neg_mean_squared_error':neg_square, 'r2':r2},
                                    return_estimator=True, n_jobs=self._n_jobs, return_indices=True, error_score='raise')

                        else: # If not AlphaGPR
                            store['metrics'] = cross_validate(regr, cv_X, loc_y, cv=GroupKFold(cv),
                                    groups=self.groups, return_train_score=True, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                    return_estimator=True, n_jobs=self._n_jobs, return_indices=True, error_score='raise')

                    else: # If no groups
                        n_samples = cv_X.shape[0]
                        if cv > n_samples:
                            print('Reducing CV-folds to ' + str(n_samples) + ', since dataset only has that many unique samples')
                            cv = n_samples
                        if isinstance(model, AlphaGPR):
                            loc_y = pd.concat([y[obj], std[obj+'_std']], axis=1)
                            ## Custom scorers to strip away the deviations we pass through along with y (so the pipeline doesn't rescale them during CV_scoring)
                            def r2(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return r2_score(y[:,0], y_pred)
                            def neg_square(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return - mean_squared_error(y[:,0], y_pred)
                            def neg_abs(rgr, X, y):
                                y_pred = rgr.predict(X)
                                if isinstance(y, pd.DataFrame): y = np.array(y)
                                return - mean_absolute_error(y[:,0], y_pred)
                            store['metrics'] = cross_validate(regr, cv_X, loc_y, cv=cv,
                                    return_train_score=True, scoring={'neg_mean_absolute_error':neg_abs, 'neg_mean_squared_error':neg_square, 'r2':r2},
                                    return_estimator=True, n_jobs=self._n_jobs, return_indices=True, error_score='raise')
                        else:
                            store['metrics'] = cross_validate(regr, cv_X, loc_y, cv=cv,
                                    return_train_score=True, scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                    return_estimator=True, n_jobs=self._n_jobs, return_indices=True, error_score='raise')
                    if verbose:
                        print("{} performance for objective function: {}".format(regr_name, obj))
                        print("{} performance for objective function: {}".format(regr_name, obj))
                        #display(store['metrics'])
                        display(pd.DataFrame({key:store['metrics'][key] for key in [
                            "fit_time", "score_time", 
                            "test_neg_mean_absolute_error", "train_neg_mean_absolute_error", 
                            "test_neg_mean_squared_error", "train_neg_mean_squared_error", 
                            "test_r2", "train_r2"]
                            }))

    def poly_lr_coefs(self, estimator_name: str) -> Dict[str, pd.DataFrame]:
        """
        Method to output polynomial coefficients and deviations for linear regression models. The coefficients listed are the coefficients for the model fit on the full training dataset, whereas the STD values (ddof=1) are based on the standard deviations of the coefficients for the cross-validated fits.

        Parameters
        ----------
        estimator_name: str
            Name of trained / cross validated estimator from workflow.

        Returns
        -------
            Dict[str, pd.DataFrame]
                Dictionary of DataFrames with keys:str - objective function names. The DataFrames have the following columns:

                ========== ============ ===== ========= ========
                Poly Index Poly Degrees Coefs Coef STDs Rel STDs
                ========== ============ ===== ========= ========

                Where Poly Degrees is in the form:

                ========= ========= === =========
                Feature_1 Feature_2 ... Feature_n
                Poly deg. Poly deg. ... Poly deg.
                ========= ========= === =========

                Poly deg. is the power of the input feature corresponding feature for the generated polynomial feature, and Rel STDs is abs(Coef STD / Coefficient)
        """
        objective_funcs = list(self.y.columns)

        results = {i:None for i in objective_funcs}

        for obj, models in self.results.items():
            if not estimator_name in models.keys(): continue
            if not isinstance(self.get_estimator(estimator_name, obj), PolynomialRegression):
                raise Exception("This method only accepts Polynomial Regression estimators")
            values = models[estimator_name]
            if not values['metrics']:
                raise Exception("Error, need to run wf.cross_validate first!")
            ## CV_pipelines : our list of trained polynomial regression pipelines based on the CV folded training data from wf.cross_validate
            CV_pipelines = values['metrics']['estimator']

            coef_table = []
            poly_unscaled = np.array(self.poly_unscaled)
            regr_features = self.get_estimator(estimator_name, obj).get_params()['polynomials']
            poly_table = [self.poly_transformer.powers_[i] for i in regr_features]
            feature_std = [np.std(poly_unscaled[:,i], ddof=1) for i in regr_features]
            for pipe in CV_pipelines:
                regr = pipe[1]
                coef_table.append([*[regr.intercept_], *(regr.coef_/feature_std)])
            coef_table = pd.DataFrame(coef_table)
            coefs = pd.Series(coef_table.mean(), name='coefs')
            stds = pd.Series(coef_table.std(ddof=1), name='std')
            stds.fillna(0.)
            rel_stds = pd.Series(np.abs(stds/coefs), name='rel_std')
            power_df = pd.concat([pd.DataFrame(np.zeros(self.X.shape[1], dtype=int)).T, pd.DataFrame(poly_table)], axis=0).reset_index(drop=True)
            power_df.columns = self.X.columns
            poly_index = pd.Series([*['*'], *regr_features], name='poly_index')
            results[obj] = pd.concat([poly_index, power_df, coefs, stds, rel_stds], axis=1).set_index('poly_index')
        
        return results


    def mean_cv_scores(self, objective_funcs: Optional[Union[str, List[str]]] = None, cv: Optional[int] = 5, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Method to calculate the mean scores and Standard Error of the Mean (SEM) of WorkFlow models.
        The metrics calculated are: time, MAE train, MAE test, MSE train, MSE test, R2 train, R2 test.
        Where test/train declares whether the prediction scoring is on the training or test slice of the dataset.
        The scores are returned as a Dict of DataFrames with an entry for each objective function.
        The scores are also stored in the WorkFlow scoring database.

        Parameters
        ----------
        objective_funcs: Optional[Union[str, List[str]]]
            str or List[str] of objective functions (string names) for which to calculate the mean cross validated metric scores.
            If ``None`` all objective functions for the WorkFlow will be calculated.

            Default value ``None``.

        cv: Optional[int]
            Number of cross validation folds to use (only relevant if models not already cross-val-scored).

            Default value ``5``.

        verbose: bool
            Whether to print out the scores.

            Default value ``True``.

        Returns
        -------
            Dict[str, pd.DataFrame]
                Dictionary of DataFrames with keys:str - objective function names. The DataFrames have the following columns:

                ==== ======== === =======
                time time_sem ... ..._sem
                ==== ======== === =======

                E.g. each metric and the Standard Error of the Mean is returned as a column.

                .. math:: np.std(metric)/np.sqrt(n_{samples})

        """
        if objective_funcs == None: objective_funcs = list(self.y.columns)
        #Auto-run cv just in case
        self.cross_validate(cv=cv, objective_funcs=objective_funcs, verbose=verbose)
        scores = {}
        for obj, models in self.results.items():
            if not obj in objective_funcs: continue
            result_dict = {}
            for name, values in models.items():
                if not values['metrics']: continue#skip models we haven't yet scored
                n_samples = len(values['metrics']['test_neg_mean_squared_error'])
                result_dict[name] = {
                        'time': np.mean(values['metrics']['fit_time']),
                        'time_sem': np.std(values['metrics']['fit_time'], dtype=np.float64)/np.sqrt(n_samples),
                        'MAE train': -1*np.mean(values['metrics']['train_neg_mean_absolute_error']),
                        'MAE train_sem': np.std(values['metrics']['train_neg_mean_absolute_error'], dtype=np.float64)/np.sqrt(n_samples),
                        'MAE test': -1*np.mean(values['metrics']['test_neg_mean_absolute_error']),
                        'MAE test_sem': np.std(values['metrics']['test_neg_mean_absolute_error'], dtype=np.float64)/np.sqrt(n_samples),
                        'MSE test': -1*np.mean(values['metrics']['test_neg_mean_squared_error']),
                        'MSE test_sem': np.std(values['metrics']['test_neg_mean_squared_error'], dtype=np.float64)/np.sqrt(n_samples),
                        'MSE train': -1*np.mean(values['metrics']['train_neg_mean_squared_error']),
                        'MSE train_sem': np.std(values['metrics']['train_neg_mean_squared_error'], dtype=np.float64)/np.sqrt(n_samples),
                        'R2 test': 1 - max(0., np.mean(values['metrics']['test_r2'])),# negative r2 scores go to zero
                        'R2 test_sem': min(0.5, np.std(values['metrics']['test_r2'], dtype=np.float64))/np.sqrt(n_samples),# if  variance > 0.5 then r2: error
                        'R2 train': 1 - max(0, np.mean(values['metrics']['train_r2'])),
                        'R2 train_sem': min(0.5, np.std(values['metrics']['train_r2'], dtype=np.float64))/np.sqrt(n_samples),
                        }
                ## IMPORTANT NOTE: I'm taking 1-R2 here so that low -> better general trend for graphic

            scores[obj] = pd.DataFrame(result_dict).sort_values('MSE test', axis=1)
        return scores

    def autoML(self, k_fold:int = 5, verbose=True) -> None:
        """
        Train a set of baseline models on the dataset and output info on best performer.

        The set of baseline models are chosen depending on the number 
        and shape of the input data:

            if dataset_size < 500: 
                "iRBF", "iMatern" (read: GPR with isotropic kernel)

                if input feature dimensions > 1, additionally use: "aRBF", "aMatern" (read: GPR with anisotropic kernel)

                if the data was given with objective_std information, additionally use:
                "..._alpha" for all aforementioned GPR models (use :class:`.AlphaGPR`)

            elif dataset_size < 1000:
                use GPR models as above, in addition, train 3 :class:`.PolynomialRegression` models:
                "PR 1", "PR 2", "PR 3" (read: model trained on polynomial features up to nth degree)
                and train one random forest model with scikit-learn default hyperparameters: "RF"

            elif dataset_size < 5000:
                exclude GPR models, use aforementioned PR and RF models and, in addition, use a MLPRegressor with lbfgs solver: "NN"

            else: 
                Use a MLPRegressor with lbfgs solver: "NN"


        Parameters
        ----------
        k_fold: int
            Number of folds to use for CV scoring. If scoring isn't set to ``None`` this parameter is moot.

            Default value ``5``

        verbose: bool
            Whether to output information on the fitting process (models used by autoML, best performing model overview).

            Default value ``True``

        Returns
        -------
            ``None``
        """
        dataset_size = self.X.shape[0]
        if dataset_size < 500:
            if verbose: print('Small dataset (n_data < 500), using exclusively GPR models, can manually include other models with WorkFlow.add_regr(\'<name>\', \'<type>\')')
            # Default simple GPR
            self.add_regr("iRBF", "gpr_RBF_iso")
            self.add_regr("iMatern", "gpr_RBF_iso")
            # Only try anisotropic kernel if n_dim > 1
            if self.X.shape[1] > 1:
                self.add_regr("aRBF", "gpr_RBF_aniso")
                self.add_regr("aMatern", "gpr_Matern_aniso")

            #Noise-Optimized GPR, only usable if we pass objective_std information
            if isinstance(self.std, pd.DataFrame):
                for obj in self.y.columns:
                    if obj+"_std" in self.std.columns:
                        self.add_regr("iRBF_alpha", "gpr_RBF_iso", obj, infer_alpha=True)
                        self.add_regr("iMatern_alpha", "gpr_Matern_iso", obj, infer_alpha=True)
                        # Only try anisotropic kernel if n_dim > 1
                        if self.X.shape[1] > 1:
                            self.add_regr("aMatern_alpha", "gpr_Matern_aniso", obj, infer_alpha=True)
                            self.add_regr("aRBF_alpha", "gpr_RBF_aniso", obj, infer_alpha=True)

        elif dataset_size < 1000:
            if verbose: print('Small dataset (n_data < 1000), excluding neural network models, can manually include other models with WorkFlow.add_regr(\'<name>\', \'<type>\')')
            # Default simple GPR
            self.add_regr("iRBF", "gpr_RBF_iso")
            self.add_regr("iMatern", "gpr_RBF_iso")
            # Only try anisotropic kernel if n_dim > 1
            if self.X.shape[1] > 1:
                self.add_regr("aRBF", "gpr_RBF_aniso")
                self.add_regr("aMatern", "gpr_Matern_aniso")

            #Noise-Optimized GPR, only usable if we pass objective_std information
            if isinstance(self.std, pd.DataFrame):
                for obj in self.y.columns:
                    if obj+"_std" in self.std.columns:
                        self.add_regr("iRBF_alpha", "gpr_RBF_iso", obj, infer_alpha=True)
                        self.add_regr("iMatern_alpha", "gpr_Matern_iso", obj, infer_alpha=True)
                        # Only try anisotropic kernel if n_dim > 1
                        if self.X.shape[1] > 1:
                            self.add_regr("aMatern_alpha", "gpr_Matern_aniso", obj, infer_alpha=True)
                            self.add_regr("aRBF_alpha", "gpr_RBF_aniso", obj, infer_alpha=True)

            # Default simple LR 1-max deg
            for degree in range(1, self.poly_transformer.degree+1):
                self.add_regr("PR "+str(degree)+"deg", "poly_lr", degree=degree)

            # Default, untuned random forest
            self.add_regr("RF", "rf")

        elif dataset_size < 5000:
            if verbose: print('Medium dataset (1000 < n_data < 5000), excluding GPR, can manually include other models with WorkFlow.add_regr(\'<name>\', \'<type>\')')

            # Default simple LR 1-max deg
            for degree in range(1, self.poly_transformer.degree+1):
                self.add_regr("PR "+str(degree)+"deg", "poly_lr", degree=degree)

            # Default, untuned random forest
            self.add_regr("RF", "rf")

            # Default, untuned neural network
            self.add_regr("NN", "nn", solver='lbfgs', max_iter=2000)

        else:
            # Default, untuned neural network
            self.add_regr("NN", "nn", solver='lbfgs', max_iter=1000)

        #Score models and return info on the best performer TODO: Refactor this
        model_scores = self.mean_cv_scores(cv=k_fold, verbose=False)
        obj_model_dict = {}
        for obj, df in model_scores.items():
            obj_model_dict[obj] = df.columns[0] #first column has lowest test MSE score i.e. best
        for obj, model in obj_model_dict.items():
            metrics = model_scores[obj]
            if verbose:
                print('Best performing model for objective ' + obj +': ' + model)
                print('CV average model score on test (unseen) data MSE: {}, R2: {}'.format(
                    metrics.at['MAE test', model],
                    1-metrics.at['R2 test', model]))
                print('Model performance on validation set:')
                self.validate(model, obj)

    def hyperparameter_optimize(self,
            fit_name: str, regr_name: str, objective_funcs: Optional[Union[str, List[str]]] = None,
            verbose: bool = True, scoring: Optional[Callable[..., float]] = None, k_fold: int = 5,
            **opt_params
        ) -> None:
        """
        Automated Bayesian Optimization of regression model hyperparameters using the GPyOpt library.

        This method encapsulates 4 different optimization methods:

            For ``regr_name="rf"``:
                The `GPyOpt <http://github.com/SheffieldML/GPyOpt>`_ library is used to perform
                Bayesian hyperparameter-optimization for a scikit-learn random-forest regression model.
                This method explores the following hyperparameter dimensions for the architecture which scores best
                with `k-fold` cross-validation:

                ``"min_samples_split" range(2,20)``

                ``"min_samples_leaf" range(1,10)``

                ``"max_depth" range(1,31,5)``

                ``"n_estimators" range(100,2200,300)``

                ``"max_features" (0.1,1.0)``


            For ``regr_name="nn"``:
                The `GPyOpt <http://github.com/SheffieldML/GPyOpt>`_ library is used to perform
                Bayesian hyperparameter-optimization for a scikit-learn MLPRegressor model.
                This method explores the following hyperparameter dimensions for the architecture which scores best
                with `k-fold` cross-validation:

                ``"hidden_layer_1" range(0,20)``

                ``"hidden_layer_2" range(0,20)``

                ``"hidden_layer_3" range(0,20)``

                ``"hidden_layer_4" range(0,20)``

                ``"alpha" (0.0001,0.01)``

                ``"batch_size" range(10,200,5)``

                ``"solver" ['lbfgs', 'sgd', 'adam']``

                ``"activation" ['identity', 'logistic', 'tanh', 'relu']``

                ``"max_iter" range(500,5001,500)``

            For ``regr_name="poly_lr"``:
                Polynomial features up to the max degree as definied during WorkFlow initialization are recursively eliminated
                by estimating the training error of the :class:`.PolynomialRegression` model for an infinite number of training
                data as a function of used polynomial features. This is done by extrapolating the linear trend of training error
                as a function of (1/N_training_data) at x=0. This error(N_inf) value is scored for models trained on each set of
                polynomials excluding one, and the pool of polynomials with the lowest error(N_inf) is selected for running the
                algorithm again. The stopping condition is, by default, the point where error(N_inf) exceeds a 10\% increase
                from the minimum error. The reduced set of polynomials are then saved as the optimized model.
                This method is based on a similar approach used in
                `previous work <https://doi.org/10.1002/cmtd.202200008>`_.

            For ``regr_name="lasso_lr"``
                The training data is fit with a scikit-learn LassoCV model with polynomial features up to the
                max degree as definied during WorkFlow initialization. The polynomials which are eliminated
                from the Lasso model are tracked, and a :class:`PolynomialRegression` model is saved with matching
                polynomial features.


        Parameters
        ----------
        fit_name: str
            Unique name under which the model is saved. If later another training is done with the same name, it will overwrite

        regr_name: str
            Which regression model to use. See: WorkFlow.supported_models
            In addition: ridge_lr, lasso_lr can be selected.

        objective_funcs: Optional[Union[str, List[str]]]

            Default value ``None``.

        verbose: bool
            Toggles whether to output information on optimization process
            
            Default value ``True``.

        scoring: Optional[Callable[..., float]]
            If callable, signature ``scorer(estimator, X, y) -> float``, otherwise, if ``None``, k-fold cross-validation optimizing MSE.

            Default value ``None``.

        k_fold: int
            Number of folds to use for CV scoring. If scoring isn't set to ``None`` this parameter is moot.

            Default value ``5``.

        \*\*opt_params: kwargs
            Accepts kwargs parameters for bayesian optimization algorithm.

        Returns
        -------
            ``None``
        """
        if not regr_name in self.supported_models:
            print(" \'{}\' invalid regression model, try one of: {}".format(regr_name, self.supported_models))
            return

        ## Set the random state for the GPyOpt hyperparameter optimization
        np.random.seed(self._random_state)

        ## Start with the end -- this dictionary will hold the optimized hyperparameters found by the search
        hyper_opt = {}

        ## Default to all objective functions if none are provided
        if objective_funcs == None: objective_funcs = list(self.y.columns)
        ## Default scoring function: 5-fold MSE
        if scoring == None:
            def scoring(regr, X, y):
                if hasattr(self, 'groups'):
                    score = np.array(cross_val_score(regr, X, y, cv=GroupKFold(k_fold), groups=self.groups, scoring='neg_mean_squared_error', n_jobs=self._n_jobs).mean())
                else:
                    score = np.array(cross_val_score(regr, X, y, cv=k_fold,scoring='neg_mean_squared_error', n_jobs=self._n_jobs).mean())
                return score if ~np.isnan(score) else -np.inf
        
        # Some useful local variables
        X = self.X
        y = self.y
        poly_X = self.poly_X
        objective_list = list(y.columns)
        feature_list = list(X.columns)
        
        #General GpyOpt optimization process:: Takes parameters for gpyopt, returns dictionary of optimal parameters
        def opt(
             f=None,
             domain=None,
             constraints=None,
             model_type='GP',
             acquisition_type='EI',
             acquisition_jitter=0.05,
             exact_feval=False,
             num_cores=1, #set this for reproducibility
             maximize=True,
             max_iter=500,
             max_time=120,
             eps=0.001,
             verbosity=verbose):
            if verbose: print("Loading optimizer for {}".format(fit_name))
            optimizer = GPyOpt.methods.BayesianOptimization(f=f,
                    domain=domain,
                    constraints=constraints,
                    model_type=model_type,
                    acquisition_type=acquisition_type,
                    acquisition_jitter=acquisition_jitter,
                    exact_feval=exact_feval,
                    num_cores=num_cores,
                    maximize=maximize)
            if verbose: print("Running hyperparameter optimization")
            optimizer.run_optimization(max_iter=max_iter,
                    max_time=max_time,
                    eps=eps,
                    verbosity=verbosity)
            if verbose:
                hyperdict = mapper(optimizer.x_opt)
                #hyperdict = {}
                #for num in range(len(domain)):
                #    hyperdict[domain[num]["name"]] = optimizer.x_opt[num]
                print("Finished hyperparameter optimization, best score: {}\nWith hyperparameters: {}".format(optimizer.fx_opt, hyperdict))
                optimizer.plot_convergence()
            return optimizer.x_opt
            
        
        ## Run the optimization for each objective in the list of objective funcs given
        for obj in to_list(objective_funcs):
            if regr_name == 'nn':
                ## Pull the activation function out as it's actually a parameter for our NN, not gpyopt
                #activation = opt_params.pop('activation') if 'activation' in opt_params.keys() else 'relu'
                opt_params['domain'] = [{'name': 'hidden_layer_1', 'type': 'discrete', 'domain': range(0,20)},
                                            {'name': 'hidden_layer_2', 'type': 'discrete', 'domain': range(0,20)},
                                            {'name': 'hidden_layer_3', 'type': 'discrete', 'domain': range(0,20)},
                                            {'name': 'hidden_layer_4', 'type': 'discrete', 'domain': range(0,20)},
                                            {'name': "alpha", 'type': 'continuous', 'domain': (0.0001,0.01)},
                                            {'name': "batch_size", 'type': 'discrete', 'domain': (10,200,5)},
                                            {'name': "solver", 'type': 'discrete', 'domain': range(0,3)},
                                            {'name': "activation", 'type': 'discrete', 'domain': range(0,4)},
                                            {'name': 'max_iter', 'type': 'discrete', 'domain': range(500,5001,500)}]
                opt_params['constraints'] = [{'name': 'constraint_on_layer_2', 'constraint': 'x[:, 1] - x[:, 1] * x[:, 0] - 1e-08'},
                                                 {'name': 'constraint_on_layer_3', 'constraint': 'x[:, 2] - x[:, 2] * x[:, 1] - 1e-08'},
                                                 {'name': 'constraint_on_layer_4', 'constraint': 'x[:, 3] - x[:, 3] * x[:, 2] - 1e-08'}]
                #Local function to convert the output of the GPyOpt hyperparameters into the sklearn formatting
                def mapper(x):
                    solvers = ['lbfgs', 'sgd', 'adam']
                    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
                    local_hyperparameters = {'hidden_layer_1' : int(x[0]),
                                             'hidden_layer_2' : int(x[1]),
                                             'hidden_layer_3' : int(x[2]),
                                             'hidden_layer_4' :int(x[3]),
                                             'alpha' : float(x[4]),
                                             'batch_size' : int(x[5]),
                                             'solver' : solvers[int(x[6])],
                                             'activation' : activation_functions[int(x[7])],
                                             'max_iter' : int(x[8]),
                                             #'activation': activation,
                                             }

                    # convert hidden layers into a tuple without zeroes
                    hidden_layer_sizes = []
                    for size in [local_hyperparameters.pop('hidden_layer_1'),
                                 local_hyperparameters.pop('hidden_layer_2'),
                                 local_hyperparameters.pop('hidden_layer_3'),
                                 local_hyperparameters.pop('hidden_layer_4')]:
                        if size > 0: hidden_layer_sizes.append(size)
                    local_hyperparameters['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
                    # values are now mapped to a format the sklearn estimator can take
                    return local_hyperparameters

                # Function called by optimizer, passes the hyperparameters to try as dict x, returns score to be optimized
                def f(x):
                    hyperparameters = mapper(x[0])# x[0] since gpyopt passes a nested numpy array [[values]]
                    regr = MLPRegressor(**hyperparameters, random_state=self._random_state)
                    score = scoring(regr, X, y[obj])
                    if verbose:
                        print(hyperparameters, "\nScore: {}".format(score))
                    return score

                opt_params['f'] = f
                hyper_opt = mapper(opt(**opt_params))

            elif regr_name == 'rf':
                opt_params['domain'] =  [{'name': "min_samples_split", 'type': 'discrete', 'domain': range(2,20)},
                        {'name': "min_samples_leaf", 'type': 'discrete', 'domain': range(1,10)},
                        {'name': 'max_depth', 'type': 'discrete', 'domain': range(1,31,5)},
                        {'name': 'n_estimators', 'type': 'discrete', 'domain': range(100,2200,300)},
                        {'name': "max_features", 'type': 'continuous', 'domain': (0.1,1.)}]

                def mapper(x):
                    return {'min_samples_split' : int(x[0]),
                              'min_samples_leaf' : int(x[1]),
                              'max_depth' : int(x[2]),
                              'n_estimators' : int(x[3]),
                              'max_features' : float(x[4])}
                def f(x):
                    hyperparameters = mapper(x[0])# x[0] since gpyopt passes a nested numpy array [[values]]
                    regr = RandomForestRegressor(**hyperparameters, random_state=self._random_state, n_jobs=self._n_jobs)
                    score = scoring(regr, X, y[obj])
                    if verbose:
                        print(hyperparameters, "\nScore: {}".format(score))
                    return score

                opt_params['f'] = f
                hyper_opt = mapper(opt(**opt_params))

            elif regr_name.startswith('gpr_'):
                if regr_name.endswith('Matern_iso'):
                    opt_params['domain'] = [{'name': "length_scale", 'type': 'continuous', 'domain': (1e-05,1e4)},
                            {'name': "nu", 'type': 'discrete', 'domain': (0.5, 1.5, 2.5, np.inf)},
                            {'name': "noise_level", 'type': 'continuous', 'domain': (1e-05, 1e5)}]

                    def mapper(x):
                        return {'kernel':  Matern(length_scale=float(x[0]), nu=float(x[1])) + WhiteKernel(noise_level=float(x[2]), noise_level_bounds='fixed')}

                elif regr_name.endswith('RBF_iso'):
                    opt_params['domain'] = [{'name': "length_scale", 'type': 'continuous', 'domain': (1e-05,1e4)},
                            {'name': "noise_level", 'type': 'continuous', 'domain': (1e-05, 1e5)}]

                    def mapper(x):
                        return {'kernel':  1*RBF(length_scale=float(x[0])) + WhiteKernel(noise_level=float(x[1]), noise_level_bounds='fixed')}

                elif regr_name.endswith('RQ_iso'):
                    opt_params['domain'] = [{'name': "length_scale", 'type': 'continuous', 'domain': (1e-05,1e4)},
                            {'name': "alpha", 'type': 'continuous', 'domain': (1e-05,1e4)},
                            {'name': "noise_level", 'type': 'continuous', 'domain': (1e-05, 1e5)}]

                    def mapper(x):
                        return {'kernel':  1*RationalQuadratic(length_scale=float(x[0]), alpha=float(x[1])) + WhiteKernel(noise_level=float(x[2]), noise_level_bounds='fixed')}

                elif regr_name.endswith('Matern_aniso'):
                    opt_params['domain'] = [{'name': "nu", 'type': 'discrete', 'domain': (0.5, 1.5, 2.5, np.inf)},
                            {'name': "noise_level", 'type': 'continuous', 'domain': (1e-05, 1e5)}]
                    for feature in feature_list:
                        opt_params['domain'].append({'name': feature + "_length", 'type': 'continuous', 'domain': (1e-05,1e4)})

                    def mapper(x):
                        return {'kernel':  1*Matern(length_scale=x[2:], nu=float(x[0])) + WhiteKernel(noise_level=float(x[1]), noise_level_bounds='fixed')}

                elif regr_name.endswith('RBF_aniso'):
                    opt_params['domain'] = [{'name': "noise_level", 'type': 'continuous', 'domain': (1e-05, 1e5)}]
                    for feature in feature_list:
                        opt_params['domain'].append({'name': feature + "_length", 'type': 'continuous', 'domain': (1e-05,1e4)})

                    def mapper(x):
                        return {'kernel':  1*RBF(length_scale=x[1:]) + WhiteKernel(noise_level=float(x[0]), noise_level_bounds='fixed')}

                def f(x):
                    hyperparameters = mapper(x[0])# x[0] since gpyopt passes a nested numpy array [[values]]
                    regr = GaussianProcessRegressor(**hyperparameters, random_state=self._random_state, n_restarts_optimizer=10)
                    score = scoring(regr, X, y[obj])
                    if verbose:
                        print(hyperparameters, "\nScore: {}".format(score))
                    return score

                opt_params['f'] = f
                hyper_opt = mapper(opt(**opt_params))

            elif regr_name == 'lasso_lr':
                poly_transformer = self.poly_transformer
                model = LassoCV(cv=10, random_state=self._random_state).fit(poly_X, y[obj]) 
                polynomials = []
                for i, coef in enumerate(model.coef_):
                    if coef != 0.: polynomials.append(i)
                nonzero_coefficients = poly_transformer.powers_[model.coef_ != 0.,:]
                hyper_opt = {'polynomials': polynomials}
                if verbose:
                    print("{}:\n{} polynomials selected:".format(
                        obj,
                        nonzero_coefficients.shape[0]))
                    display(pd.DataFrame(nonzero_coefficients, columns=X.columns).reset_index(drop=True))

            elif regr_name == 'ridge_lr':
                poly_transformer = self.poly_transformer
                model = RidgeCV(cv=10, scoring='neg_mean_squared_error').fit(poly_X, y[obj]) 
                polynomials = []
                for i, coef in enumerate(model.coef_):
                    if coef != 0.: polynomials.append(i)
                nonzero_coefficients = poly_transformer.powers_[model.coef_ != 0.,:]
                hyper_opt = {'polynomials': polynomials}
                if verbose:
                    print("{}:\n{} polynomials selected:".format(
                        obj,
                        nonzero_coefficients.shape[0]))
                    display(pd.DataFrame(nonzero_coefficients, columns=X.columns).reset_index(drop=True))

            elif regr_name == 'poly_lr':
                # Transform our input to the polynomial degree desired
                #default parameters

                ## Select polynomials up to a degree such that the number of features is less than half the training set
                initial_polynomials = []
                X = self.poly_X
                poly_table = self.poly_transformer.powers_

                ## Confusing name: elim_min_degree corresponds to minimum degree polynomial must have to be considered for elimination
                elim_min_degree = self.poly_transformer.degree if not 'degree' in opt_params.keys() else opt_params['degree']
                for p in range(len(poly_table)-1,0,-1):
                        if np.sum(poly_table[p]) == elim_min_degree:
                            if p <= X.shape[0]/2:
                                initial_polynomials = [*range(p+1)]
                                break
                            else:
                                print("Warning, too few data to analyze degree {}, trying lower order polynomials.".format(elim_min_degree))
                                elim_min_degree -= 1

                params = dict(n_trials = 20, max_deterioration = 10, elim_min_degree = elim_min_degree, initial_polynomials = initial_polynomials)
                # overwrite defaults with explicit declared params
                for key, val in opt_params.items():
                    params[key] = val
                #this one separate since it's computationally intensive (calc default value only if needed)
                params['max_poly_performance'] = params['max_poly_performance'] if 'max_poly_performance' in params.keys() else self._datasize_performance(params['initial_polynomials'], obj)[0]

                polynomials = self._recursive_eta2_minimize(params['n_trials'],
                        obj,
                        params['max_deterioration'],
                        params['elim_min_degree'],
                        params['max_poly_performance'],
                        params['initial_polynomials'],
                        verbose=verbose)

                # All that for this: Optimization == best performing polynomials selected
                hyper_opt = {'polynomials': polynomials}

            # Finally, add the model to our regression list
            self.add_regr(fit_name, regr_name, obj, **hyper_opt)


    def __str__(self):
        return "LECA regression model training object\nSupported models: " + str(self.supported_models) + "\nSupported metrics: " + str(self.supported_metrics)
    
    def get_estimator(self,
            estimator_name: str, objective_funcs: Optional[Union[str, List[str]]] = None
        ) -> Union[BaseEstimator, List[BaseEstimator], None]:
        """
        Returns the named fitted estimator object(s) (if exists) for objective function(s).

        Parameters
        ----------
        estimator_name: str
            Unique model identifier name.

        objective_funcs: Optional[Union[str, List[str]]]
            String or list of strings enumerating the objective functions for which to get estimators.
            If ``None``, all objective functions will have the associated named model returned.

        Returns
        -------
            Union[BaseEstimator, List[BaseEstimator]]
                If a singular str ``objective_funcs`` is passed, the named estimator object is returned, otherwise a list of the estimator objects corresponding to the List[str] of ``objective_funcs`` is returned.
        """
        if objective_funcs == None: objective_funcs = list(self.y.columns)

        estimator_list = []
        for obj in to_list(objective_funcs):
            try:
                estimator_list.append(self.results[obj][estimator_name]['model'])
            except KeyError:
                print("No estimator under the name \'{}\' for {} found".format(estimator_name, obj))
        return estimator_list[0] if len(estimator_list) == 1 else estimator_list

    def polynomial_convert(self,
            X: pd.DataFrame, X_scaled: bool = False
        ) -> pd.DataFrame:
        """
        Takes unscaled or scaled input DataFrame X (must match the DataFrame format used in WorkFlow) and transforms it into a scaled polynomial to match with the expected input for polynomial regression models.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame of input feature vectors, matching the DataFrame format for WorkFlow.

        X_scaled: bool
            Whether DataFrame X has scaled features (``True``) or not.

            Default value ``False``.

        Returns
        -------
            pd.DataFrame
                DataFrame of scaled polynomial features matching the format for polynomial regression models with this WorkFlow.
        """
        if X_scaled: X = self.scaler.inverse_transform(X)
        return pd.DataFrame(self.poly_scaler.transform(self.poly_transformer.transform(X)))

    def best_model(self,
            objectives: Optional[Union[str, List[str]]] = None
            ) -> Union[str, Dict[str, str]]:
        """
        Return best scoring model name(s) (MSE) for objective(s).

        Parameters
        ----------
        objectives: Optional[Union[str, List[str]]]
            Optional parameter to define which models to return. If none, the best scoring model for each objective is returned.

        Returns
        -------
            Union[str, Dict[str, str]]
                If a single objective is passed, the name of the best model is returned. If a list is passed, a dictionary in the form {'objective_name':'model_name} is returned.
        """
        if objectives == None:
            best_models = {}
            model_scores = self.mean_cv_scores(verbose=False)
            for obj, df in model_scores.items():
                best_models[obj] = df.columns[0] #first column has lowest test MSE score i.e. best

        # If we pass a single string as our objective, convert single {obj:model} dict with best scoring model
        if isinstance(objectives, str):
            obj = objectives # redundant but for readability
            best_model = self.mean_cv_scores(verbose=False)[obj].columns[0]
            best_models = best_model

        if isinstance(objectives, list):
            for obj in objectives:
                best_model = self.mean_cv_scores(verbose=False)[obj].columns[0]
                best_models[obj] = best_model

        return best_models


    def predict(self,
            X: pd.DataFrame, objectives: Optional[Union[str,Dict[str, str]]] = None, X_scaled: bool = False,
            min_max: bool = False, return_std: bool = False
        ) -> pd.DataFrame:
        """
        Call model to predict objective function for given input feature vectors.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame of input feature vectors, matching the DataFrame format for WorkFlow.

        objectives: Optional[Union[str,Dict[str, str]]]
            Dictionary with format:

            ============== ===================
            Obj fn 1 name: Model A string name
            Obj fn 2 name: Model B string name
            ============== ===================

            If a string is passed, the best scoring model will be selected for the objective function matching with the string.

            If ``None`` the model with the best mean cross-validated MSE score for each objective function in WorkFlow is selected.

            Default value ``None``.

        X_scaled: bool
            Whether DataFrame X has scaled features (``True``) or not.

            Default value ``False``.

        min_max: bool
            Whether to use the `min_max` method to estimate uncertainty, or MAPIE with conformity scores.
            `min_max`:``True`` takes the minimum and maximum prediction of the bootstrapped models to be the range of uncertainty.
            `min_max`:``False`` uses the MAPIE uncertainty estimation outlined in: `Mapie jackknife+-AB <https://mapie.readthedocs.io/en/latest/theoretical_description_regression.html#the-jackknife-after-bootstrap-method>`_
            This parameter is moot for GPR models.

            Default value ``False``.

        return_std: bool
            Whether to return also the uncertainty estimation for predictions.
            If bootstrapped models not yet trained, will automatically call ``WorkFlow.estimate_uncertainty()`` for the selected models.

            Default value ``False``.

        Returns
        -------
            pd.DataFrame
                DataFrame with the following conditional named columns of predictions (and their one-sigma uncertainty):

                ============= ============= ===
                Obj fn 1 name Obj fn 2 name ...
                ============= ============= ===

                If return_std = ``True``:

                ============= ================= === =======
                Obj fn 1 name Obj fn 1 name_std ... ..._std
                ============= ================= === =======
        """
        if not len(X.columns) == len(self.X.columns):
            raise Exception("Dataset must be given as a labeled dataframe (including the following columns: {})".format(list(self.X.columns)))
        if not all(X.columns == self.X.columns): X = X.reindex(columns = self.X.columns) # Auto-order to match wf.X
        if X.dropna().shape[1] < len(X.columns):# if the columns in the given X are missing any values they're filled by NaN with reindex. This catches that case
            raise Exception("Dataset must be given as a labeled dataframe (including the following columns: {}\n (or in very edge case, the input X values may not include NaNs))".format(list(self.X.columns)))

        if objectives == None:
            objectives = {}
            model_scores = self.mean_cv_scores(verbose=False)
            for obj, df in model_scores.items():
                objectives[obj] = df.columns[0] #first column has lowest test MSE score i.e. best

        # If we pass a single string as our objective, convert single {obj:model} dict with best scoring model
        if isinstance(objectives, str):
            obj = objectives # redundant but for readability
            best_model = self.mean_cv_scores(verbose=False)[obj].columns[0]
            objectives = {obj : best_model}

        prediction = pd.DataFrame()
        for obj, estimator_name in objectives.items():
            ## Check estimator exists
            if not estimator_name in self.results[obj]:
                print("Warning: {} not found for {}, skipping".format(estimator_name, obj))
                continue
            store = self.results[obj][estimator_name]
            estimator = store['model']


            local_X = X # A bit wonky, but defining a local X for each objective function allows the possibility of mixing polynomial and other regression predictions together

            ## If estimator polynomial -> polynomial convert
            if isinstance(estimator, PolynomialRegression):
                local_X = self.polynomial_convert(X, X_scaled=X_scaled)
            ## elif unscaled, scale
            elif not X_scaled:
                local_X = self.scaler.transform(X)
            # X now has proper form for estimator

            if not return_std:
                y = estimator.predict(local_X)
                prediction = pd.concat([prediction, pd.Series(y, name=obj)], axis=1)
                continue

            ## Check if it has uncertainty values
            uncert = store['uncertainty']
            ## Auto estimate uncertainty
            if not uncert:
                print("One moment - calculating uncertainty estimations")
                self.estimate_uncertainty(estimator_name, obj)
                uncert = store['uncertainty']
            if isinstance(estimator, GaussianProcessRegressor):
                #y, _ = estimator.predict(local_X, return_std=True) # prediction based on our estimator trained on full data
                y, std = estimator.predict(local_X, return_std=True) # prediction based on our estimator trained on full data
                #std = 0
                #for estimator in store['metrics']['estimator']: # sigma based on average of sigma over CV models
                #    _, loc_std = estimator.predict(local_X, return_std=True)
                #    std += loc_std

                #estimator_count = len(store['metrics']['estimator'])
                #std = std/estimator_count
            else:
                ## Mapie predictions
                ## min_max takes the difference between the minimum and maximum of our bootstrapped models as our uncert
                if min_max:
                    pred_multi = uncert._pred_multi(local_X)
                    y, std = [pred.mean() for pred in pred_multi], [np.std(pred, ddof=1) for pred in pred_multi]
                    # assume min-max range corresponds to +-sigma
                    #y, std = [pred.mean() for pred in pred_multi], [(pred.max() - pred.min())/2 for pred in pred_multi]

                ##otherwise use MAPIE Jackknife+-ab (i.e. include conformity scores penalty)
                else:
                    y, std = uncert.predict(local_X, alpha=0.32)# -> 68% confidence interval == 1*sigma ASSUMING normal distribution
                    std = np.abs(std[:,1]-std[:,0]).ravel()
            prediction = pd.concat([prediction, pd.Series(y, name=obj), pd.Series(std, name=obj+'_std')], axis=1)

        prediction.index = X.index # cast prediction indices to match the indices of the X input

        return prediction

    def arrhenius_cross_validate(self,
            original_objective: str, df: pd.DataFrame, beta_0: float,
            models: Optional[List[str]]=None,
            arrhenius_objectives: Optional[List[str]]=['S0', 'S1', 'S2'],
            save_loc: Union[bool, str] = False,
            show_title: bool = False,
            log: bool = True, deviate_by_salt: bool = True, custom_label = None,
            highlight_extrema = False
        ) -> Dict[str, float]:
        """
        Output validation plots and prediction error scores for regression models after back-transforming 
        Arrhenius surrogate model to original objective function.
        
        Uses the WorkFlow's **CV folds** to score the performance of each CV-trained model on its
        corresponding test set.

        Parameters
        ----------
        original_objective: str
            String name of pre-Arrhenius surrogate model objective function.

        df: pd.DataFrame
            DataFrame of pre-Arrhenius surrogate model measurement data.

        beta_0: float
            beta_0 value used in Arrhenius fits. See: prep.arrhenius() for more information.

        models: Optional[List[str]]
            Optionally define which models (by string name identifier) to use to predict
            the Arrhenius coefficients. The list ordering matches the objective_funcs.
            If ``None`` the best scoring model (MSE) is used by default.
            
            Default value ``None``

        objective_funcs: Optional[List[str]]
            List of string names of 3-objective functions pass to the Arrhenius function
            in the form ['S0', 'S1', 'S2']

            Default value ``['S0', 'S1', 'S2']``

        save_loc : Union[bool, str]
            Name to save plot (if desired), if ``False`` the plot will only be shown, not saved.

            Saving filename convention is:
            save_loc + objective function + '-arrhenius-cross-validate.pdf'

        log : Optional[bool]
            Whether to compare to logarithmic conductivity.
            
            Default value ``True``
            
        deviate_by_salt : Optional[bool]
            Whether log(conductivity/x_LiSalt) should be plotted
            
            Default value ``True``

        Returns
        -------
            Dict[str, float]:

                Dictionary of mean-CV-scores with keys (sem: std of mean): 
                    "r2_train", "r2_train_sem", "MAE_train", "MAE_train_sem", 
                    "MSE_train", "MSE_train_sem", "RMSE_train",
                    "RMSE_train_sem", "r2_test", "r2_test_sem", 
                    "MAE_test", "MAE_test_sem", "MSE_test", "MSE_test_sem",
                    "RMSE_test", "RMSE_test_sem"

        """
        # If no models explicitly defined, use best performing by MSE
        if models == None: models = list(self.best_model(arrhenius_objectives).values())
        # Local vars for string names of each arrhenius coefficient
        s0, s1, s2 = arrhenius_objectives
        s0_model, s1_model, s2_model = models

        # Predict the conductivity using the passed CV-trained models (cv_pipelines)
        def arrh_predict(x_input, cv_pipelines, log, deviate_by_salt):
            temp_offset = np.array(x_input['inverse temperature'] - beta_0)
            x_input = x_input[self.features] #drops inverse temp (+ any other extraneous features) 

            # Predict coefficient values from each CV-pipeline corresponding to S0, S1 and S2
            pred = {s0:[],s1:[],s2:[]}
            for pipeline, objective in zip(cv_pipelines,arrhenius_objectives):
                model = pipeline[1] #the pipelines are simply scaler -> model
                if not isinstance(model, PolynomialRegression):
                    pred[objective] = pipeline.predict(x_input)
                else:
                    pred[objective] = pipeline.predict(self.poly_transformer.transform(x_input))
            #to compare log(conductivity/x_LiSalt)
            if deviate_by_salt == True:
                cond = pred[s0] - pred[s1]*(temp_offset) - pred[s2]*(temp_offset)**2
            #to compare log conductivity
            else:
                cond = pred[s0] + np.log10(x_input['x_LiSalt']) - pred[s1]*(temp_offset) - pred[s2]*(temp_offset)**2
            #to compare conductivity
            if log == False:
                cond = np.power(10,cond) 
            return pd.DataFrame({original_objective:cond})

        # make np.array: back_transform[self.X_unscaled index] -> raw conductivity measurement df with matching input features
        X = self.X_unscaled
        n_training_data = X.shape[0]
        back_transform = np.empty(n_training_data, dtype=object)
        for j in range(n_training_data):
            composition = X.iloc[j]
            composition_match = df.loc[(df[X.columns] == composition).all(axis=1)]
            back_transform[j] = composition_match

        # For each CV fold we score the fold-training and fold-test sets.
        # We add the test set predictions to the validation plot
        # And track the resulting performance metrics
        s0_data = self.results[s0][s0_model]['metrics']
        s1_data = self.results[s1][s1_model]['metrics']
        s2_data = self.results[s2][s2_model]['metrics']

        # don't plot each CV fold directly, just track the predictions, we'll plot at the end
        all_validation_predictions = pd.DataFrame()
        results = {"r2_train": [],"r2_test": [], "MAE_train": [], "MAE_test": [],
                "MSE_train": [], "MSE_test": [], "RMSE_train": [], "RMSE_test": []}

        for i in range(len(self.results[s0][s0_model]['metrics']['estimator'])):

            cv_pipelines = [
                    s0_data['estimator'][i],
                    s1_data['estimator'][i],
                    s2_data['estimator'][i]
                    ]
            
            #check that all validation slices are uniform:
            if not ((s0_data['indices']['test'][i] == s1_data['indices']['test'][i]).all() and (s2_data['indices']['test'][i] == s1_data['indices']['test'][i]).all()):
                raise ValueError('The validation slices for each CV-model differ when they should be identical. Something has gone wrong in the cross-validation step. Please report this bug to the package maintainers.')

            # Ok, we can go on the assumption now that the test/train slices are identical
            test_indices = s0_data['indices']['test'][i]
            train_indices = s0_data['indices']['train'][i]

            # Build the training/validation set with back_transform[index] -> measurement_df
            training_set  = pd.concat(back_transform[train_indices])
            validation_set = pd.concat(back_transform[test_indices])
            
            ## Calculate performance on validation set
            ## If we have validation data, plot it. Otherwise just training
            predict = arrh_predict(validation_set, cv_pipelines, log, deviate_by_salt)
            predict.index = validation_set.index
            validation_set["pred_" + original_objective] = predict
            x = validation_set[original_objective].values.flatten()
            y = predict[original_objective].values.flatten()
            all_validation_predictions = pd.concat([all_validation_predictions, validation_set])


            MAE = mean_absolute_error(np.asarray(x), np.asarray(y))
            MSE = mean_squared_error(np.asarray(x), np.asarray(y))
            
            MAE_test = mean_absolute_error(np.asarray(x), np.asarray(y))
            MSE_test = mean_squared_error(np.asarray(x), np.asarray(y))
            r2_test = r2_score(x,y)
            
            ## Also show our predictions for training data
            train_predict = arrh_predict(training_set, cv_pipelines, log, deviate_by_salt)
            train_x = training_set[original_objective].values.flatten()
            train_y = train_predict[original_objective].values.flatten()

            MAE_train = mean_absolute_error(np.asarray(train_x), np.asarray(train_y))
            MSE_train = mean_squared_error(np.asarray(train_x), np.asarray(train_y))
            r2_train = r2_score(train_x,train_y)


            #print("Validation set:\nMAE: {}\nMSE: {}".format(MAE,MSE))
            results["r2_train"].append(r2_train)
            results["r2_test"].append(r2_test)
            results["MAE_train"].append(MAE_train)
            results["MAE_test"].append(MAE_test)
            results["MSE_train"].append(MSE_train)
            results["MSE_test"].append(MSE_test)
            results["RMSE_train"].append(np.sqrt(MSE_train))
            results["RMSE_test"].append(np.sqrt(MSE_test))


        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05)
        if custom_label==None:
            title = original_objective +": "
            ax.set_ylabel('{} predicted'.format(original_objective), fontsize=18)
            ax.set_xlabel('measured {}'.format(original_objective), fontsize=18)
        else:
            title = custom_label + ": "
            ax.set_ylabel('{} predicted'.format(custom_label), fontsize=18)
            ax.set_xlabel('measured {}'.format(custom_label), fontsize=18)

        title = title + r"r$^2$=" + str(np.round(np.mean(results["r2_test"]),3))
        if show_title == True:
            ax.set_title(title, fontsize=20)

        #plot validation predictions, first just plot all
        ax.scatter(all_validation_predictions[original_objective], all_validation_predictions["pred_"+original_objective], s=25, color='black', alpha=0.5)
        # overlay red circle on max_boundary values if we declared a min extrema count to highlight
        if isinstance(highlight_extrema, int) and highlight_extrema != 0:
            sum_extrema = (
                    (all_validation_predictions[X.columns] == all_validation_predictions[X.columns].min()).sum(axis=1)      #min extrema count
                    + (all_validation_predictions[X.columns] == all_validation_predictions[X.columns].max()).sum(axis=1))   #+ max
            extrema = all_validation_predictions.loc[sum_extrema >= highlight_extrema]
            ax.scatter(extrema[original_objective], extrema["pred_"+original_objective], s=25, marker='o', facecolors='none', edgecolors='red',alpha=0.5)
        #ax.legend(loc='upper left', fontsize=18)
        #ax.tick_params(labelsize=16)
        # plot 1:1 line for perfect predictions
        ax.axline([all_validation_predictions[original_objective].min(), all_validation_predictions[original_objective].min()], slope=1, linestyle='dashed', alpha=0.3, color='black')
        plt.tight_layout()
        if save_loc: plt.savefig(save_loc +'-'+ original_objective.replace("/", "-") 
                +'-'+  s0_model +'-'+ s1_model +'-'+ s2_model +'-'+ 'arrhenius-cross-validate.pdf')
        plt.show()

        n_samples = len(results['r2_train'])
        return {
                "r2_train": np.mean(results["r2_train"]),
                "r2_train_sem": np.std(results["r2_train"], dtype=np.float64)/np.sqrt(n_samples),
                "MAE_train": np.mean(results["MAE_train"]),
                "MAE_train_sem": np.std(results["MAE_train"], dtype=np.float64)/np.sqrt(n_samples),
                "MSE_train": np.mean(results["MSE_train"]),
                "MSE_train_sem": np.std(results["MSE_train"], dtype=np.float64)/np.sqrt(n_samples),
                "RMSE_train": np.mean(results["RMSE_train"]),
                "RMSE_train_sem": np.std(results["RMSE_train"], dtype=np.float64)/np.sqrt(n_samples),
                "r2_test": np.mean(results["r2_test"]),
                "r2_test_sem": np.std(results["r2_test"], dtype=np.float64)/np.sqrt(n_samples),
                "MAE_test": np.mean(results["MAE_test"]),
                "MAE_test_sem": np.std(results["MAE_test"], dtype=np.float64)/np.sqrt(n_samples),
                "MSE_test": np.mean(results["MSE_test"]),
                "MSE_test_sem": np.std(results["MSE_test"], dtype=np.float64)/np.sqrt(n_samples),
                "RMSE_test": np.mean(results["RMSE_test"]),
                "RMSE_test_sem": np.std(results["RMSE_test"], dtype=np.float64)/np.sqrt(n_samples),
                }


    def arrhenius_validate(self,
            original_objective: str,
            df: pd.DataFrame,
            beta_0: float,
            models: List[str]=None, arrhenius_objectives: Optional[List[str]]=['S0', 'S1', 'S2'], save_loc: Union[bool, str] = False,
            log: bool = True,
            deviate_by_salt: bool = True,
            show_title: bool = True,
            custom_label = None
        ) -> Dict[str, float]:
        """
        Output validation plots and prediction error scores for regression models after back-transforming 
        Arrhenius surrogate model to original objective function.
        
        Uses the WorkFlow's **validation holdout dataset** to identify validation-set 
        compositions in the passed DataFrame.

        Parameters
        ----------
        original_objective: str
            String name of pre-Arrhenius surrogate model objective function.

        df: pd.DataFrame
            DataFrame of pre-Arrhenius surrogate model measurement data.

        beta_0: float
            beta_0 value used in Arrhenius fits. See: prep.arrhenius() for more information.

        models: Optional[List[str]]
            Optionally define which models (by string name identifier) to use to predict
            the Arrhenius coefficients. The list ordering matches the objective_funcs.
            If ``None`` the best scoring model (MSE) is used by default.
            
            Default value ``None``

        objective_funcs: Optional[List[str]]
            List of string names of 3-objective functions pass to the Arrhenius function
            in the form ['S0', 'S1', 'S2']

            Default value ``['S0', 'S1', 'S2']``

        save_loc : Union[bool, str]
            Name to save plot (if desired), if ``False`` the plot will only be shown, not saved.

            Saving filename convention is:
            save_loc + objective function + '-arrhenius-validate.pdf'

        log : Optional[bool]
            Whether to compare to logarithmic conductivity.
            
            Default value ``True``
            
        deviate_by_salt : Optional[bool]
            Whether log(conductivity/x_LiSalt) should be plotted
            
            Default value ``True``

        Returns
        -------
            Dict[str, float]:

                Dictionary of model accuracy scores with keys: 
                    "r2_train", "r2_test", "MAE_train", "MAE_test", 
                    "MSE_train", "MSE_test", "RMSE_train", "RMSE_test"
        """
        if not hasattr(self, 'X_validate'):
            print("No validation set defined")
            return

        def arrh_predict(x_input, log, deviate_by_salt):
            temp_offset = np.array(x_input['inverse temperature'] - beta_0)
            x_input = x_input[self.features]
            # By default take highest scoring models
            if models == None:
                pred = self.predict(x_input, min_max=True, return_std=True)
            else: 
                pred = self.predict(x_input, dict(zip(arrhenius_objectives, models)), min_max=True, return_std=True)

            s0, s1, s2 = arrhenius_objectives[0], arrhenius_objectives[1], arrhenius_objectives[2]
            #to compare log(conductivity/x_LiSalt)
            if deviate_by_salt == True:
                cond = (unumpy.uarray(pred[s0], pred[s0+'_std']) 
                            - unumpy.uarray(pred[s1], pred[s1+'_std'])*(temp_offset)
                            - unumpy.uarray(pred[s2], pred[s2+'_std'])*(temp_offset)**2)
            #to compare log conductivity
            else:
                cond = (unumpy.uarray(pred[s0]+np.log10(x_input['x_LiSalt']), pred[s0+'_std']) 
                            - unumpy.uarray(pred[s1], pred[s1+'_std'])*(temp_offset)
                            - unumpy.uarray(pred[s2], pred[s2+'_std'])*(temp_offset)**2)
            #to compare conductivity
            if log == False:
                cond = unumpy.pow(10,cond) #this must be changed compared to the original function 
            return pd.DataFrame({original_objective:unumpy.nominal_values(cond),
                             original_objective+'_std':unumpy.std_devs(cond)})

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        if custom_label==None:
            title = original_objective +": "
        else:
            title = custom_label + ": "

        # Build the training/validation set by cross-referencing the WorkFlow dataframe
        # compositions against the compositions in the given df
        training_set = pd.DataFrame()
        X = self.X_unscaled
        for i in range(X.shape[0]):
            composition = X.iloc[i]
            composition_match = df.loc[(df[X.columns] == composition).all(axis=1)]
            training_set  = pd.concat([training_set, composition_match])

        # We don't simply take the inverse here as some values may have been filtered out
        # before the dataset is given to the WorkFlow. This ensures a 1:1 match of
        # raw measurement data and validation set
        validation_set = pd.DataFrame()
        X = self.X_validate_unscaled
        for i in range(X.shape[0]):
            composition = X.iloc[i]
            composition_match = df.loc[(df[X.columns] == composition).all(axis=1)]
            validation_set = pd.concat([validation_set, composition_match])


        ## Calculate performance on validation set
        ## If we have validation data, plot it. Otherwise just training
        predict = arrh_predict(validation_set, log, deviate_by_salt)
        x = validation_set[original_objective].values.flatten()
        y = predict[original_objective].values.flatten()
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x),np.array(y))
        line = slope*np.array(x)+intercept

        MAE = mean_absolute_error(np.asarray(x), np.asarray(y))
        MSE = mean_squared_error(np.asarray(x), np.asarray(y))
        
        MAE_test = mean_absolute_error(np.asarray(x), np.asarray(y))
        MSE_test = mean_squared_error(np.asarray(x), np.asarray(y))
        r2_test = r2_score(x,y)
        
        
        title = title + r"r$^2$=" + str(np.round(r2_score(x,y),3))
        ## Also show our predictions for training data
        train_predict = arrh_predict(training_set, log, deviate_by_salt)
        train_x = training_set[original_objective].values.flatten()
        train_y = train_predict[original_objective].values.flatten()

        MAE_train = mean_absolute_error(np.asarray(train_x), np.asarray(train_y))
        MSE_train = mean_squared_error(np.asarray(train_x), np.asarray(train_y))
        r2_train = r2_score(train_x,train_y)

        ax.scatter(train_x, train_y, label='Training', color='black', alpha=0.5, s=25)
        ax.scatter(x, y, label='Validation', color='red', alpha=0.5, s=25)
        ax.axline([train_x[0],train_x[0]], slope=1, linestyle='dashed', alpha=0.5, color='black')
        if custom_label == None:
            ax.set_ylabel('{} predicted'.format(original_objective), fontsize=18)
            ax.set_xlabel('measured {}'.format(original_objective), fontsize=18)
        else:
            ax.set_ylabel('{} predicted'.format(custom_label), fontsize=18)
            ax.set_xlabel('measured {}'.format(custom_label), fontsize=18)

        if show_title == True:
            ax.set_title(title, fontsize=20)
        #ax.legend(loc='upper left', fontsize=18)
        #ax.tick_params(labelsize=16)

        plt.tight_layout()
        if save_loc: plt.savefig(save_loc + original_objective.replace("/", "-") + 'arrhenius-validate.pdf')
        plt.show()

        print("Validation set:\nMAE: {}\nMSE: {}".format(MAE,MSE))
        return {"r2_train": r2_train,"r2_test": r2_test, "MAE_train": MAE_train, "MAE_test": MAE_test, 
            "MSE_train": MSE_train, "MSE_test": MSE_test, "RMSE_train": np.sqrt(MSE_train), "RMSE_test": np.sqrt(MSE_test)}



    def validate(self,
            name:str, objective_funcs=None, save_loc: Union[bool, str] = False, show_title: bool = True
        ) -> None:
        """
        Output validation plots and r2 scores for regression models using stored validation dataset.

        Parameters
        ----------
        name: str
            String name of regression model.

        objective_funcs: Optional[Union[str, List[str]]]
            String name, or list of string names of objective function to score on validation dataset.

        save_loc : Union[bool, str]
            Name to save plot (if desired), if ``False`` the plot will only be shown, not saved.

            Saving filename convention is:
            save_loc + objective function + '-unseen-validate.pdf'

        Returns
        -------
            ``None``
        """
        if not hasattr(self, 'X_validate'):
            print("No validation set defined")
            return

        if objective_funcs == None: objective_funcs = list(self.y.columns)
        obj_model_dict = {x:name for x in to_list(objective_funcs)}
        train_predict = self.predict(self.X, obj_model_dict, X_scaled=True)
        n_obj = len(to_list(objective_funcs))

        fig, ax = plt.subplots(1, n_obj, figsize=(4*n_obj, 4))
        
        for i, obj in enumerate(to_list(objective_funcs)):
            title = obj +": "
            loc_ax = ax[i] if n_obj > 1 else ax
            
            train_x = self.y[obj].values.flatten()
            train_y = train_predict[obj].values.flatten()
            
            ## Also show our predictions for training data
            loc_ax.scatter(train_x, train_y, label='Training', color='black', alpha=0.5, s=25)
            loc_ax.set_ylabel('{} predicted'.format(obj), fontsize=18)
            loc_ax.set_xlabel('{} truth'.format(obj), fontsize=18)
            
            ## Calculate performance on validation set
            ## If we have validation data, plot it. Otherwise just training
            try:
                predict = self.predict(self.X_validate, obj_model_dict, X_scaled=True)
                x = self.y_validate[obj].values.flatten()
                y = predict[obj].values.flatten()
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x),np.array(y))
                line = slope*np.array(x)+intercept
                MAE = mean_absolute_error(np.asarray(x), np.asarray(y))
                MSE = mean_squared_error(np.asarray(x), np.asarray(y))
                loc_ax.scatter(x, y, label='Validation', color='red', alpha=0.5, s=25)
                #loc_ax.plot(x, line)#, label="r$^2$={}\nMAE={}\nMSE={}".format(np.round(r_value,3), np.round(MAE,3), np.round(MSE,5)))
                title = title + r"r$^2$=" + str(np.round(r2_score(x,y),3))
            finally:
                loc_ax.axline([train_x[0],train_x[0]], slope=1, linestyle='dashed', alpha=0.5, color='black')
                
            if show_title == True:
                loc_ax.set_title(title, fontsize=20)
                
        plt.tight_layout()
        if save_loc: plt.savefig(save_loc + obj.replace("/", "-") + '-unseen-validate.pdf', bbox_inches='tight')
        plt.show()

        print("Validation set:\nMAE: {}\nMSE: {}".format(MAE,MSE))
        return {"r2": r2_score(x,y), "MAE": MAE, "MSE": MSE, "RMSE": np.sqrt(MSE)}

    def _datasize_performance(self, polynomials, objective, test_size=1, N_min=None, sample_count=5, repeat=100):
        """
        1/N Performance Metric

        :polynomials: list int

        Indices of polynomials to use for PolynomialRegression

        :returns: float: N_inf estimated error, ufloat: test-error for N_max training
        """
        X = self.poly_X
        y = self.y[objective]
        if len(polynomials) == 0:
            raise Exception('Polynomial performance analaysis doesn\'t accept empty lists of polynomials')
        estimator = PolynomialRegression(polynomials=polynomials)
        #print(estimator)

        """ OLD
        total_data_count = X.shape[0]
        """
        #NEW
        if hasattr(self, 'groups'):
            total_data_count = len(self.groups.unique())
        else:
            total_data_count = X.shape[0]
        # max(1, ...) assures we never have a test size of 0
        test_count = max(1, test_size if isinstance(test_size, int) else int(test_size*total_data_count))
        N_max = total_data_count - test_count

        """ OLD VERSION
        ## FIRST: find N_min by starting with N=1 and iterating N += 1 until training-error > 0, this defines our N_min (the max N where E_training = 0)
        E_training = 0
        if N_min == None: N_min=len(polynomials)
#        N_min = 0
#        while np.isclose(E_training, 0, atol=1e-10) and N_min < N_max:
#            N_min += 1
#            X_train, y_train = X.iloc[:N_min], y.iloc[:N_min]
#            estimator.fit(X_train, y_train)
#            E_training = mean_squared_error(y_train,estimator.predict(X_train))


        ## Define training_data_count = [N_min,...,N_max] so len(training_data_count) = sample_count  with values which are equidistant on a 1/N scale 
        training_data_count = np.array(1/np.linspace(1/N_min, 1/N_max, sample_count), dtype=int)

        results_dict = {}
        for _ in range(repeat):
            ## Create a shuffled list of indices to drop into our bins:
            shuffled_list = random.sample(range(N_max), N_max)
            test_set = shuffled_list[:test_count]
            rest_set = shuffled_list[test_count:]
            X_test, y_test = X.iloc[test_set], y.iloc[test_set]
        """
        # New version
        ## N_min user defined or dimensions of feature set, if poly-lr, we use the number of polynomials as min
        if N_min == None: N_min=len(polynomials)
            
        results_dict = {}
        for _ in range(repeat):
            ## Create a shuffled list of indices to drop into our bins:

            # We use different methods if we've defined composition groups
            # composition groups -> split by group
            # otherwise, split datapoints
            if hasattr(self, 'groups'):
                shuffled_list = random.sample(list(self.groups.unique()), N_max)
                # A little tricky here, we create a boolean map for each datapoint
                # If datapoint belongs to group in the test set -> True, else false
                test_set = list(self.groups.loc[
                        [(group_id in shuffled_list[:test_count]) for group_id in self.groups]
                        ].index)
                rest_set = list(self.groups.loc[
                        [(group_id not in shuffled_list[:test_count]) for group_id in self.groups]
                        ].index)
            else:
                shuffled_list = random.sample(range(N_max), N_max)
                test_set = shuffled_list[:test_count]
                rest_set = shuffled_list[test_count:]
            X_test, y_test  = X.iloc[test_set], y.iloc[test_set]

            ## Define training_data_count = [N_min,...,N_max] so len(training_data_count) = sample_count  with values which are equidistant on a 1/N scale 
            training_data_count = np.array(1/np.linspace(1/N_min, 1/len(rest_set), sample_count), dtype=int)
            for i, endpoint in enumerate(training_data_count):
                #Initialize dict key if doesn't yet exist
                if not i in results_dict.keys():
                    results_dict[i] = {'data_size': [], 'test': [], 'train': []}
                    
                #Define bin indices and slice out the data
                this_bin = rest_set[:endpoint]
                X_train, y_train = X.iloc[this_bin], y.iloc[this_bin]
         
                #Train estimator then test
                estimator.fit(X_train, y_train)
                train_error = mean_squared_error(y_train,estimator.predict(X_train))
                test_error = mean_squared_error(y_test,estimator.predict(X_test))
                #Save results
                results_dict[i]['train'].append(train_error)
                results_dict[i]['test'].append(test_error)
                results_dict[i]['data_size'].append(endpoint)
        # Take average score and store the std
        for endpoint, score_dict in results_dict.items():
            score_dict['test_std'] = np.std(score_dict['test'], ddof=0)/np.sqrt(repeat)#TODO check this!
            score_dict['test'] = np.mean(score_dict['test'])
            score_dict['train_std'] = np.std(score_dict['train'], ddof=0)/np.sqrt(repeat)#TODO check this!
            score_dict['train'] = np.mean(score_dict['train'])
            score_dict['data_size'] = np.mean(score_dict['data_size'])

        df = pd.DataFrame(results_dict).T.set_index('data_size').T
        lin_fit = LinearRegression().fit(np.array(1/df.columns).reshape(-1, 1), df.loc['train'])
        eta_squared = lin_fit.intercept_
        slope = lin_fit.coef_[0]
        #print("Estimated eta**2: {}".format(eta_squared))
        #if isinstance(estimator, PolynomialRegression):
        #    print("P_eff estimated to be {}".format(-slope/eta_squared))
        return eta_squared, ufloat(df.loc['test'].iloc[-1],df.loc['test_std'].iloc[-1])

    def _recursive_eta2_minimize(self, n_trials,objective,max_deterioration,min_degree,max_poly_performance,current_features, verbose=False):
        """
        Automatic recursive polynomial selection via elimination of polynomials with minimal increase on eta2 score.
        """
        test_performance = []
        poly_table = self.poly_transformer.powers_
        
        # check performance for each feature set excluding current selected feature
        current_features.sort(reverse=True)
        for feature in current_features:
            if np.sum(poly_table[feature]) < min_degree: break # if the total degree of the polynomial feature under the min, we're done here
            test_features = current_features.copy()
            test_features.remove(feature)
            test_features.sort()
            try:
                test_performance.append(self._datasize_performance(test_features, objective, repeat=n_trials)[0])
            except:
                break
        
        delta = -np.inf
        best_performance = np.inf
        if len(test_performance) > 0:
            minimal_impact = np.argmin(test_performance)
            ## If removing a feature deteriorates performance by over 10% of starting performance, stop searching
            irrelevant_feature = current_features[minimal_impact]
            best_performance = test_performance[minimal_impact]
            delta = (max_poly_performance-best_performance)/max_poly_performance*100
        if delta < -max_deterioration:
            if min_degree <= 1:
                current_features.sort()
                if verbose: print("Choosing: {}".format(current_features))
                return current_features
            else:
                if verbose: print("\nAll features with degree {} appear relevant, searching degree {}.\n".format(min_degree, min_degree-1))
                best_performance = np.min((max_poly_performance,best_performance))
                return self._recursive_eta2_minimize(n_trials,objective,max_deterioration,min_degree-1,
                                     max_poly_performance,
                                     current_features,verbose)
        ## Else, continue recursion
        else:
            current_features.remove(irrelevant_feature)
            best_performance = np.min((max_poly_performance,best_performance))
            if verbose: print("Irrelevant feature found: {} ({}), N_inf error delta: {}%\nCurrent best err_N_inf: {}".format(
                irrelevant_feature,
                poly_table[irrelevant_feature],
                delta,
                best_performance))
            return self._recursive_eta2_minimize(n_trials,objective,max_deterioration,min_degree,
                                      max_poly_performance,
                                      current_features,verbose)

    def optimize(self, strategy:str = 'max',
            obj_fn:Optional[Callable] = None,
            fixed_values:Optional[Dict[str,float]] = None,
            bounds:Optional[Dict[str,Tuple[float]]] = None,
            n_restarts_optimizer:int = 100) -> pd.DataFrame:
        """
        Optimizer to search design space for max/min objective value, bayesian expected improvement, upper/lower confidence bound and maximum uncertainty strategies. Returns optimal input feature set to query for given strategy.

        Parameters
        ----------
        strategy: str
            Optimization strategy to use:

                - ``max`` : Maximize obj_fn
                - ``min`` : Minimize obj_fn
                - ``EI`` : Maximize bayesian expected improvement
                - ``UCB`` : Maximize upper confidence bound (obj_fn+std)
                - ``LCB`` : Minimize lower confidence bound -(obj_fn+std)
                - ``max_uncert`` : Maximize obj_fn uncertainty

            Default value ``max``.

        obj_fn: Optional[Callable f(x: pd.DataFrame) -> pd.DataFrame]
            Callable function which takes a feature DataFrame input (using the same features as the workflow) and returns a 2x1 dataframe in with columns ['objective', 'objective_std'].

            Default value ``None``

        fixed_values: Optional[Dict[str,float]]
            Dictionary of fixed {'feature name' : value}s for optimization task.

            Default value ``None``

        bounds: Optional[Dict[str,Tuple[float]]]
            Dictionary of {'feature name' : (min, max)} for setting the boundaries to search for optimization task.

            Default value ``None``

        n_restarts_optimizer: int
            Number of random points in the design space from which the acquisition function will be optimized. Higher -> more computationally expensive, but higher chance of finding global best acquisition point.

            Default value ``100``.
        
        Returns
        -------
            pd.DataFrame
                DataFrame of input features and objective prediction
        """
        supported_strats = ['max', 'min', 'EI', 'UCB', 'LCB', 'max_uncert']
        if not strategy in supported_strats: raise Exception("Unsupported strategy. Try one of: {}".format(supported_strats))

        # Avoid error checking for values in fixed_values.keys()
        if fixed_values == None: fixed_values = {}
        if bounds == None: bounds = {}

        ## Define search bounds
        for feature in self.X.columns:
            if feature in fixed_values.keys(): continue
            if feature in bounds.keys(): continue
            bounds[feature] = (self.X_unscaled[feature].min(), self.X_unscaled[feature].max())
            

        ## If we don't pass a custom objective function, default to objective functions of WorkFlow
        if not callable(obj_fn):
            objectives = self.y.columns
            result = pd.DataFrame()

            # Calculate optimal sample for each objective function
            for obj in objectives:
                def obj_fn(x):
                    return self.predict(x, obj, X_scaled=False, min_max=True,return_std=True)

                obj_optimal_sample = self.optimize(strategy=strategy,
                                            obj_fn=obj_fn,
                                            fixed_values=fixed_values
                                            )
                result = pd.concat([result, obj_optimal_sample], axis=0)
            return result

        # df.iat[0,0] => obj
        # df.iat[0,1] => obj_std
        if strategy == 'max' or strategy == 'min': # query by committee
            def f(x):
                sample = dict(zip(bounds.keys(), x))
                for feature, val in fixed_values.items():
                    sample[feature] = val
                df_sample = pd.DataFrame(sample, index=[0]) #DataFrame with one entry
                result = obj_fn(df_sample).iat[0,0] ## <- predicted standard deviations at sample point
                return -result if strategy == 'max' else result

            res = self._optimizer(f, bounds, n_restarts_optimizer)

            optimal_sample = dict(zip(bounds.keys(), res))
            for feature, val in fixed_values.items():
                optimal_sample[feature] = val
            optimal_sample = pd.DataFrame(optimal_sample, index=[0])
            sample_result = obj_fn(optimal_sample)
            return pd.concat([optimal_sample, sample_result], axis=1)

        elif strategy == 'max_uncert':
            def f(x):
                sample = dict(zip(bounds.keys(), x))
                for feature, val in fixed_values.items():
                    sample[feature] = val
                df_sample = pd.DataFrame(sample, index=[0]) #DataFrame with one entry
                std = obj_fn(df_sample).iat[0,1] ## <- predicted standard deviations at sample point
                return -std

            res = self._optimizer(f, bounds, n_restarts_optimizer)

            optimal_sample = dict(zip(bounds.keys(), res))
            for feature, val in fixed_values.items():
                optimal_sample[feature] = val
            optimal_sample = pd.DataFrame(optimal_sample, index=[0])
            sample_result = obj_fn(optimal_sample)
            return pd.concat([optimal_sample, sample_result], axis=1)

        elif strategy == 'UCB' or strategy == 'LCB':
            def f(x):
                sample = dict(zip(bounds.keys(), x))
                for feature, val in fixed_values.items():
                    sample[feature] = val
                df_sample = pd.DataFrame(sample, index=[0]) #DataFrame with one entry
                std = obj_fn(df_sample).iat[0,1] ## <- predicted standard deviations at sample point
                result = obj_fn(df_sample).iat[0,0] ## <- predicted objective at sample point
                return -(std + result) if strategy == 'UCB' else (std + result)

            res = self._optimizer(f, bounds, n_restarts_optimizer)

            optimal_sample = dict(zip(bounds.keys(), res))
            for feature, val in fixed_values.items():
                optimal_sample[feature] = val
            optimal_sample = pd.DataFrame(optimal_sample, index=[0])
            sample_result = obj_fn(optimal_sample)
            return pd.concat([optimal_sample, sample_result], axis=1)

        elif strategy == 'EI': # expected improvement

            # Define our current maximum objective value in the training data pool

            # First define valid pool matching fixed values
            if len(fixed_values.keys()) == 0:   # If we have no fixed values, all points are valid
                data_pool = self.X_unscaled
            else:                               # Otherwise we have to slice for (nearly) matching values
                # make a boolean series checking that all X_unscaled values == corresponding fixed_value
                matching_values = self.X_unscaled[fixed_values.keys()].apply(
                        lambda x: np.isclose(x, list(fixed_values.values())).all(), axis=1
                        )
                data_pool = self.X_unscaled.loc[matching_values]

            # If pool is empty, assume ymax = 0
            if data_pool.shape[0] == 0: ymax = 0
            # Else take max value from predictions on every pool element
            else:
                ymax = obj_fn(data_pool).max()[0]

            def f(x):
                sample = dict(zip(bounds.keys(), x))
                for feature, val in fixed_values.items():
                    sample[feature] = val
                df_sample = pd.DataFrame(sample, index=[0]) #DataFrame with one entry
                pred = obj_fn(df_sample)
                # m: median, s: sigma
                m, s = pred.iat[0,0], pred.iat[0,1]
                u = (m - ymax) / s
                ei = s * (u * st.norm.cdf(u) + st.norm.pdf(u))
                ei = 0. if s <= 0. else ei
                return -ei

            res = self._optimizer(f, bounds, n_restarts_optimizer)

            optimal_sample = dict(zip(bounds.keys(), res))
            for feature, val in fixed_values.items():
                optimal_sample[feature] = val
            optimal_sample = pd.DataFrame(optimal_sample, index=[0])
            sample_result = obj_fn(optimal_sample)
            return pd.concat([optimal_sample, sample_result], axis=1)

    def _optimizer(self, f, bounds, n_restarts_optimizer:int = 100):
        min_list = []
        # Brute force here to avoid local minima. Generate random in-bounds x0 values to attempt minimization
        x0_array = np.array([np.random.uniform(*min_max, n_restarts_optimizer) for min_max in bounds.values()]).T
        for x0 in x0_array:
            res = opt.minimize(fun=f,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds.values()
                    )
            min_list.append(res['x'])

        solution_id = np.argmin([f(x) for x in min_list])
        solution = min_list[solution_id]
        ## Perturb one value to avoid identical sampling (testing!)
        #solution[0] = solution[0] + np.random.normal(0,1e-5)
        return solution

    def generate_bins(self,
            objective_funcs:Optional[Union[str, List[str]]] = None,
            features:Optional[Union[str, List[str]]] = None,
            fixed_values:Optional[Dict[str,float]] = None,
            min_bins:int = 3, feature_importance_bins:int = 10,
            manual_bins:Optional[Dict[str,List[float]]] = None,
            manual_min_max_bounds:Optional[Dict[str,Tuple[float]]] = None,
            validity_test:Optional[Callable] = None
            ) -> pd.DataFrame:
        """
        Method to create a DataFrame of discrete values spanning the design space. Features which show a higher importance (via random-forest feature importance metric) will have more bins. This method will either automatically determine the number of (equidistant) bins in the min-to-max range of a feature, or alternately accepts user defined bins for a feature in the form of a list of values.

        The key formula for bin generation is:

        .. math:: bin\_count = max(importance_{objective_1},...)*feature\_importance\_bins + min\_bins
        
        If automatic bin generation is used, equidistant bins from the min-to-max value range of the feature in the WorkFlow DataFrame are generated scaling with the feature's highest importance for the given ``objective_funcs``.

        Parameters
        ----------
        objective_funcs: Optional[Union[str, List[str]]]
            String or list of strings of the objective functions to be considered. If ``None``, defaults to all objective functions in the WorkFlow.

            Default value ``None``

        features: Optional[Union[str, List[str]]]
            String or list of strings of the binned features. If ``None``, defaults to all features in the WorkFlow.

            Default value ``None``

        fixed_values: Optional[Dict[str,float]]
            Dictionary of fixed {'feature name' : value} to be included in the output DataFrame (e.g. could include a fixed inverse temperature with fixed_values = {'inverse temperature':3.0}``.) NOTE: Any features included as a fixed value will be -excluded- from the random forest fits used for determining feature importance.

            Default value ``None``

        min_bins: int
            Minimum number of bins automatically assigned to a feature. Features with no measured importance from the random forest fit will have this many bins.

            Default value ``3``

        feature_importance_bins: int
            Maximum number of bins automatically assigned to a feature. Features a measured importance of ``1`` from the random forest fit will have this many bins.

            Default value ``10``

        manual_bins:Optional[Dict[str,List[float]]]
            Optional parameter to explicitly define the bins for a feature. Accepts a dictionary in the form: ``'{feature':[list of values]}``, any features not included in this dictionary will have automatically selected bins.

            Default value ``None``

        manual_min_max_bounds:Optional[Dict[str,tuple[float]]]
            Optional parameter to define the min-to-max range for a feature. Passed as a dictionary of tuples: ``'{feature':(min, max)}``, any features not included in this dictionary will take the min-to-max range from the WorkFlow training DataFrame.

            Default value ``None``

        validity_test: Optional[Callable f(x: pd.DataFrame) -> pd.DataFrame]
            Callable function which takes a feature DataFrame input (i.e. df.columns == feature_list) and returns a boolean list where ``True`` represents a valid composition, and ``False`` indicates a composition to be excluded.

            Default value ``None``

        Returns
        -------
            pd.DataFrame
                DataFrame of discrete feature values spanning the design space.
        """
        # Populate any local var if None was passed
        objective_funcs = self.objective_list if objective_funcs == None else to_list(objective_funcs) 
        features = self.features if features == None else to_list(features) 
        fixed_values = {} if fixed_values == None else fixed_values
        manual_bins = {} if manual_bins == None else manual_bins
        manual_min_max_bounds = {} if manual_min_max_bounds == None else manual_min_max_bounds

        # Generate a feature importance score for each objective function
        max_feature_importance_dict = {feature : 0. for feature in features}
        for objective_fn in objective_funcs:
            # run a simple RF fit on each objective function
            rf = RandomForestRegressor(n_jobs=self._n_jobs)
            fit_features = [ft for ft in features if ft not in fixed_values.keys()]
            rf.fit(self.X[fit_features], self.y[objective_fn])

            # Clearly associate feature_importance from rf fit with feature name
            feature_importance = dict(zip(fit_features, rf.feature_importances_))

            # overwrite max_feature_importance_dict with any feature importance > previous max
            max_feature_importance_dict = {ft : max(max_feature_importance_dict[ft],
                feature_importance[ft]) for ft in fit_features}


        feature_bin_counts = {
                ft: int(max_feature_importance_dict[ft]*feature_importance_bins + min_bins)
                for ft in max_feature_importance_dict.keys()}

        ### Define values for each bin in each feature dimension
        bin_values = {}
        for feature in features:
            if feature in manual_bins.keys():
                bin_values[feature] = manual_bins[feature]
            elif feature in fixed_values.keys():
                bin_values[feature] = to_list(fixed_values[feature])
                # else take values from min to max with a higher resolution for higher feature importance
            else:
                bin_values[feature] = np.linspace(
                        # if user defined min-max range
                        *(manual_min_max_bounds[feature] if feature in manual_min_max_bounds.keys()
                        # otherwise, take from WorkFlow dataframe
                        else (self.X_unscaled[feature].min(), self.X_unscaled[feature].max())),
                        # with a resolution weighted by feature importance
                         feature_bin_counts[feature]
                         ) 

        # Output overview of selected bins
        for feat, arr in bin_values.items():
            print('{:<10} {} bins: \t~ {:.3} to {:.3}'.format(feat, len(arr), min(arr), max(arr)))
     
        ### Create dataframe (~outer product of each 1dim array of bin values)
        pool_df = pd.DataFrame(product(*(bin_values.values())),columns=list(bin_values.keys()))
            
        ### Eliminate values which violate user defined conditions
        ### (e.g. sum(specific salt or additive fractions (PF6, FSI, etc.)) <= 1)
        if callable(validity_test): pool_df = pool_df.loc[validity_test(pool_df)]

        # Return dataframe of discrete coordinates in feature space
        return pool_df.reset_index(drop=True)

    def RBMAL(self,
            uncertainty_fn:Optional[Callable] = None,
            pool:Optional[pd.DataFrame] = None,
            batch_size:int = 10,
            target_uncertainty:Optional[float] = 0
            ) -> Tuple[List[int], pd.DataFrame]:
        """
        Method to apply Ranked Batch Mode Active Learning approach to return recommended queries.

        Adapted from modAL RBMAL implementation
        https://github.com/modAL-python/modAL/blob/7f72997b6dc26e8fe063b90d409c7cfcf4ef418e/modAL/batch.py

        Based on RBMAL approach proposed by Cardoso et al.
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

        Parameters
        ----------

        uncertainty_fn:Optional[Callable]
            Optional function which takes the query pool and returns a DataFrame of uncertainty
            values analogous to the model's prediction uncertainty. If ``None`` the prediction
            uncertainty of the workflow's highest scoring model is used. In the case of multiple
            objective functions, the automatic uncertainty calculation will take the highest
            estimated uncertainty from the predictions for each objective function.

            Default value ``None``

        pool:Optional[pd.DataFrame]
            DataFrame of candidate points in feature space to query. If ``None`` is passed, the
            `generate_bins` method is used to automatically generate a grid in feature space.

            Default value ``None``

        batch_size:int
            Number of points to query.

            Default value ``10``

        target_uncertainty:Optional[float]
            Sets lower bound on model uncertainty. If any prediction is estimated to have an
            uncertainty lower than this value that space is considered well defined and
            excluded from the candidate points to query.

            Default value ``0``

        Returns
        -------
            Tuple[List[int], pd.DataFrame]
                DataFrame of ranked points in feature space from highest to lowest priority
                for further training the ML models.
        """
        #TODO: Handle case where no more points are above uncertainty threshold
        
        scaler = MinMaxScaler()
        metric='euclidean'
        X_pool = pool if isinstance(pool, pd.DataFrame) else self.generate_bins()
        X_training = self.X_unscaled.reset_index(drop=True)

        ## Calculate uncertainty scores for each point in the candidate pool
        # If no uncertainty fn provided, take max uncertainty from all obj function predictions (best model) 
        if uncertainty_fn == None:
            # define column labels for uncertainties
            cols = [obj + '_std' for obj in self.objective_list]
            # take max uncertainty for any objective fn prediction
            X_uncertainty = self.predict(X_pool, return_std=True)[cols].max(axis=1)
        else:
            X_uncertainty = uncertainty_fn(X_pool)

        original_pool_size = X_pool.shape[0]
        #filter out points in query pool below uncertainty threshold
        uncertainty_filter = np.where(X_uncertainty > target_uncertainty)
        X_pool = X_pool.iloc[uncertainty_filter]
        X_uncertainty = X_uncertainty.iloc[uncertainty_filter]
        filtered_pool_size = X_pool.shape[0]

        if target_uncertainty > 0:
            print("{:.2f}% of candidate pool has an estimated uncertainty over the target uncertainty.".format(filtered_pool_size/original_pool_size*100))

        if filtered_pool_size <= batch_size:
            #print("Batch size > Uncertainty filtered query points")
            if filtered_pool_size > 0:
                return list(X_pool.index), X_pool
            else:
                return [], pd.DataFrame()

        #normalize uncertainty from 0 (target threshold) to 1 (max)
        normalized_uncert = (X_uncertainty - target_uncertainty) / (np.max(X_uncertainty) - target_uncertainty)

        #normalize distances in feature space
        scaler.fit(pd.concat([X_pool, X_training])) # fit scaler to rescale all features from 0 (min) : 1 (max)
        normalized_pool = scaler.transform(X_pool)
        normalized_training = scaler.transform(X_training)

        #ratio between unlabeled / labeled points for calculating alpha value
        n_unlabeled_points = X_pool.shape[0]
        n_labeled_points = X_training.shape[0]


        # Calculate pairwise distances between candidate query points and already labeled data
        if self._n_jobs == 1 or self._n_jobs is None:
            _, distance_scores = pairwise_distances_argmin_min(normalized_pool, normalized_training, metric=metric)
        else:
            distance_scores = pairwise_distances(normalized_pool, normalized_training, metric=metric,
                                                 n_jobs=self._n_jobs).min(axis=1)

        ### We have all the ingredients we need to start looping and selecting our queries
            # Distance scores
            # Normalized pool
            # labeled / unlabeled
            # normalized uncert

        queries = []
        mask = np.ones(n_unlabeled_points, dtype=bool)

        for i in range(batch_size):

            similarity_scores = 1 / (1 + distance_scores)

            alpha = (n_unlabeled_points - i) / (n_unlabeled_points + n_labeled_points)

            scores = alpha * (1 - similarity_scores) + (1 - alpha) * normalized_uncert

            # Pardon, confusing implementation: Reason - we want to keep track of the original pool indices
            # We have df_idx corresponding to original indices of the candidate pool dataframe
            # list_idx corresponds to the index of the same entry referenced by df_idx but referring to its position in the np list of scores range(0:n_unlabeled-1)
            #set the scores of our previously chosen queries to -1 so idxmax / argmax never choose an already queried point (but crucially, the score list indices aren't shifted by masking)
            scores[~mask] = -1
            top_score_df_idx = scores.idxmax()
            top_score_list_idx = scores.argmax()

            #TODO?: store these scores to output as a query overview
            #display(X_pool.iloc[top_score_idx])
            #print('sim:{}, unc:{}, sc:{}\n'.format(similarity_scores[top_score_list_idx], normalized_uncert[top_score_df_idx], scores[top_score_df_idx]))

            queries.append(top_score_df_idx)
            mask[top_score_list_idx] = False

            # We'll sample the top scoring query, so we downgrade distance scores of any other proximate query points
            distance_scores = np.minimum(distance_scores, pairwise_distances(normalized_pool,
                                                    [normalized_pool[top_score_list_idx]], metric=metric,
                                                    n_jobs=self._n_jobs).min(axis=1))

        return queries, X_pool.loc[queries]
