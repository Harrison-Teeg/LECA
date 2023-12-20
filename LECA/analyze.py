import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
## LR
from sklearn.linear_model import LinearRegression
from LECA.estimators import PolynomialRegression, AlphaGPR
## Metrics
from scipy import stats
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_score
## Bayesian Optimization
import GPy, GPyOpt
## Uncertainty handling
from uncertainties import unumpy, ufloat
## Cloning and random for 1/N plots
from sklearn.base import clone
import random
# For type annotations
from typing import List, Tuple, Union, Optional, Callable, Dict
from sklearn.base import BaseEstimator
from LECA.fit import WorkFlow
from LECA.prep import to_list
import LECA.prep as prep
# Mirko helper functions' req
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def comparative_datasize_performance(
        wf: WorkFlow, estimators: Optional[Union[str, List[str]]] = None,
        objective_funcs: Optional[Union[str, List[str]]] = None,
        test_size: Union[int, float] = 0.1, N_min: Optional[int] = None, sample_count: int = 5,
        repeat: int = 100, log_scale: bool = False, plot: bool = False, y_lim: Optional[Tuple[float]] = None,
        random_state: Optional[int] = None,
        confidence = 1.0,
        save_loc: Union[str, bool] = False
    ) -> Dict[str, Tuple[float, pd.DataFrame]]:
    """
    Comparative model performance as a function of N_training size.
    The MSE scores of the named regression model are calculated for the given objective
    functions using randomly selected datapoints for fixed fractions of the dataset to 
    analyze the model performance as a function of the number of training datapoints.

    Parameters
    ----------
    wf: WorkFlow
        WorkFlow object with models to analyze.

    estimators: Optional[Union[str, List[str]]]
        String or list with model name(s) to perform datasize performance analysis. ``None`` will use all workflow models.

        Default value ``None``.

    objective_funcs: Optional[Union[str, List[str]]]
        String or list of strings declaring which objective functions on which to perform datasize performance analysis.
        When ``None`` is passed, defaults to all objective functions.
        If a list of objective functions are passed, the function returns a list of objects.

        Default value ``None``.

    test_size: Union[int, float]
        If int: Explicit number of datapoints used as the test set.

        If float: Defines fraction of the whole dataset to use as the test set.

        Default value ``0.1``.

    N_min: Optional[int]
        Number of training datapoints for the first (smallest) sample.
        If ``None`` then the first non-zero E_training results will be used as N_min.

        Default value ``None``.

    sample_count: int
        Number of points to sample for training/prediction error.
        I.e. the number of different datasizes to score for training/test MSE.
        The sample sizes will be automatically selected as equidistant on the N_training scale on the range from N_min to N_total.

        Default value ``5``.

    repeat: int, default=100
        How many times to repeat datasize performance test.
        The mean values of the repeated analysis and their standard deviations are then recorded as the results.

        Default value ``100``.

    log_scale: bool
        Whether to plot the y axis with a log scale.

        Default value ``False``.

    plot: bool
        Whether to output a plot of the training/test MSE scores as a function of N_training.

        Default value ``False``.

    random_state : Optional[int]
        Sets a numpy random seed for reproducibility.

        Default value ``None``.
        
    confidence : float
        Set confidence intervall for error bars. confidence*standard deviation is shown as error bars.
        
        Default value ``1.0``.
        
    save_loc: Union[str, bool]
        Destination to save result plot (if provided as a string argument).
        Figure is saved to: `save_loc + 'model_compare_N_data-' + obj.replace("/", "-")  + ".pdf"`
        Where `obj` is the objective function of the model prediction.

        Default value ``False``.

    Returns
    -------
        Dict[str, Dict[str, pd.DataFrame]]
            Returns a dictionary with each key the string name of the listed objective functions.
            Each objective key has a dictionary value in the form 'model_name': results dataFrame (for that model).
            The results DataFrame has the test/train MSE and their deviations for each model's performance on different dataset slices.
            The DataFrame has the form:

            ========= ============ ============ ===
            -         slice 1 size slice 2 size ...
            test      ...          ...          ...
            train     ...          ...          ...
            test_std  ...          ...          ...
            train_std ...          ...          ...
            ========= ============ ============ ===

            Where test / train are the mean of the MSE scores on the test / train dataset for the models, and test/train_std are their deviations.
    """
    if objective_funcs == None: objective_funcs = list(wf.y.columns)
    rng = np.random.default_rng(seed=random_state)

    #useful local vars
    poly_X, X,y,std = wf.poly_X, wf.X, wf.y, wf.std

    # If we have composition groups, we split by unique comps, not by individual datapoints
    if hasattr(wf, 'groups'):
        total_data_count = len(wf.groups.unique())
    else:
        total_data_count = X.shape[0]

    test_count = max(1, test_size if isinstance(test_size, int) else int(test_size*total_data_count))
    N_max = total_data_count - test_count

    result_dict = {obj:{} for obj in to_list(objective_funcs)}
    for obj in to_list(objective_funcs):
        if estimators == None: estimators = wf.results[obj].keys()
        ## We're running a list of estimators now, not just one...
        estimators = to_list(estimators)

        ## N_min user defined or dimensions of feature set
        if N_min == None:
            N_min = X.shape[1]
            #if isinstance(estimator, PolynomialRegression): N_min = len(estimator.polynomials)
            

        results_dict = {estimator:{} for estimator in estimators}
        for _ in range(repeat):

            # We use different methods if we've defined composition groups
            # composition groups -> split by group
            # otherwise, split datapoints
            if hasattr(wf, 'groups'):
                shuffled_list = rng.choice(list(wf.groups.unique()), N_max, replace=False)
                # A little tricky here, we create a boolean map for each datapoint
                # If datapoint belongs to group in the test set -> True, else false
                test_set = list(wf.groups.loc[
                        [(group_id in shuffled_list[:test_count]) for group_id in wf.groups]
                        ].index)
                rest_set = list(wf.groups.loc[
                        [(group_id not in shuffled_list[:test_count]) for group_id in wf.groups]
                        ].index)
            else:
                shuffled_list = rng.choice(range(N_max), N_max, replace=False)
                test_set = shuffled_list[:test_count]
                rest_set = shuffled_list[test_count:]
            X_test, y_test = X.iloc[test_set], y[obj].iloc[test_set]
            poly_X_test = poly_X.iloc[test_set]

            ## Define training_data_count = [N_min,...,N_max] so len(training_data_count) = sample_count
            training_data_count = np.array(np.linspace(N_min, len(rest_set), sample_count), dtype=int)

            for i, endpoint in enumerate(training_data_count):
                for model in estimators:
                    estimator = clone(wf.get_estimator(model, obj))#copies hyperparameters but is unfitted, avoids overwriting our prefit model
                    #Initialize dict key if doesn't yet exist
                    if not i in results_dict[model].keys():
                        results_dict[model][i] = {'data_size': [], 'test': [], 'train': []}
                        
                    #Define bin indices and slice out the data
                    this_bin = rest_set[:endpoint]
                    X_train, y_train = X.iloc[this_bin], y[obj].iloc[this_bin]
                    if isinstance(estimator, PolynomialRegression):
                        X_train = poly_X.iloc[this_bin]

                    #Train estimator then test
                    if isinstance(estimator, AlphaGPR):
                        alpha_train = pd.concat([y[obj].iloc[this_bin], std[obj+"_std"].iloc[this_bin]], axis=1)
                        estimator.fit(X_train, alpha_train)
                    else:
                        estimator.fit(X_train, y_train)

                    if isinstance(estimator, PolynomialRegression):
                        test_error = mean_squared_error(y_test,estimator.predict(poly_X_test))
                    else:
                        test_error = mean_squared_error(y_test,estimator.predict(X_test))
                    train_error = mean_squared_error(y_train,estimator.predict(X_train))
                    #Save results
                    results_dict[model][i]['train'].append(train_error)
                    results_dict[model][i]['test'].append(test_error)
                    results_dict[model][i]['data_size'].append(endpoint)
        for model, result in results_dict.items():
            # Take average score and store the std
            for _, score_dict in result.items():
                score_dict['test_std'] = np.std(score_dict['test'], ddof=0)/np.sqrt(repeat)#TODO check this!
                score_dict['test'] = np.mean(score_dict['test'])
                score_dict['train_std'] = np.std(score_dict['train'], ddof=0)/np.sqrt(repeat)#TODO check this!
                score_dict['train'] = np.mean(score_dict['train'])
                score_dict['data_size'] = np.mean(score_dict['data_size'])
        
            df = pd.DataFrame(result).T.set_index('data_size').T
            if plot:
                plt.plot(figsize=(6,4))
                #plt.title('MSE to N_training',fontsize=20)
                plt.xlabel('$N_\\mathrm{training}$')
                if log_scale: 
                    plt.ylabel(r'log(MSE)')
                    plt.errorbar(df.columns,np.log10(df.loc['test']),
                            yerr=(confidence*df.loc['test_std']/df.loc['test']),#TODO: double check this is correct sigma for log scale
                            xerr=None,marker='o',label=model, alpha=1,capsize=3, markerfacecolor='white')
                    last_plot = plt.gca().lines[-1]
                    previous_color = last_plot.get_color()
                    plt.errorbar(df.columns,np.log10(df.loc['train']),
                            yerr=(confidence*df.loc['train_std']/df.loc['train']),
                            #yerr=np.log((df.loc['train']-1.96*df.loc['train_std'])/(df.loc['train']+1.96*df.loc['train_std'])),
                            xerr=None,marker='s', color=previous_color, linestyle='--', alpha=1,capsize=3, markerfacecolor='white')
                else:
                    plt.ylabel(r'$\eta^2$')
                    plt.errorbar(df.columns,df.loc['test'],yerr=(confidence*df.loc['test_std']),xerr=None,marker='o',label=model, alpha=1,capsize=3, markerfacecolor='white')
                    last_plot = plt.gca().lines[-1]
                    previous_color = last_plot.get_color()
                    plt.errorbar(df.columns,df.loc['train'],yerr=(confidence*df.loc['train_std']),xerr=None,marker='s', color=previous_color, linestyle='--', alpha=1,capsize=3, markerfacecolor='white')
            result_dict[obj][model] = df
        #display(df)
        if plot: 
            if obj+"_std" in std.columns:
                mean_deviations = std[obj+"_std"].mean()
                print("Experimental Variance: {}".format(np.power(mean_deviations,2)))
                if log_scale:
                    plt.hlines(2*np.log10(mean_deviations), 0, N_max, linestyles='-', color='black', label=r'log($\eta_{measured}^2$)')
                else:
                    plt.hlines(np.power(mean_deviations, 2), 0, N_max, linestyles='-', color='black', label=r'$\eta_{measured}^2$')
            plt.xticks()
            plt.yticks()
            plt.legend(fontsize=12)
            if y_lim != None:
                plt.ylim(y_lim)
            plt.tight_layout()
            if save_loc: plt.savefig(save_loc + 'model_compare_N_data-' + obj.replace("/", "-")  + ".pdf", bbox_inches="tight")
            plt.show()           
    return result_dict



def datasize_performance(
        wf: WorkFlow, estimator_name: str, objective_funcs: Optional[Union[str, List[str]]] = None,
        test_size: Union[int, float] = 0.1, N_min: Optional[int] = None, sample_count: int = 5,
        repeat: int = 100, plot: bool = False, random_state: Optional[int] = None,
        confidence = 1.0,
        save_loc: Union[str, bool] = False
    ) -> Dict[str, Tuple[float, pd.DataFrame]]:
    """
    1/N_training performance metrics.
    The MSE scores of the named regression model are calculated for the given objective functions using randomly selected datapoints for fixed fractions of the dataset to analyze the model performance as a function of the number of training datapoints.

    Parameters
    ----------
    wf: WorkFlow
        WorkFlow object with models to analyze.

    estimator_name: str
        String with model name to perform datasize performance analysis.

    objective_funcs: Optional[Union[str, List[str]]]
        String or list of strings declaring which objective functions on which to perform datasize performance analysis.
        When ``None`` is passed, defaults to all objective functions.
        If a list of objective functions are passed, the function returns a list of objects.

        Default value ``None``.

    test_size: Union[int, float]
        If int: Explicit number of datapoints used as the test set.

        If float: Defines fraction of the whole dataset to use as the test set.

        Default value ``0.1``.

    N_min: Optional[int]
        Number of training datapoints for the first (smallest) sample.
        If ``None`` then the first non-zero E_training results will be used as N_min.

        Default value ``None``.

    sample_count: int
        Number of points to sample for training/prediction error.
        I.e. the number of different datasizes to score for training/test MSE.
        The sample sizes will be automatically selected as equidistant on the 1/N_training scale on the range from N_min to N_total.

        Default value ``5``.

    repeat: int, default=100
        How many times to repeat datasize performance test.
        The mean values of the repeated analysis and their standard deviations are then recorded as the results.

        Default value ``100``.
        
    plot: bool
        Whether to output a plot of the training/test MSE scores as a function of 1/N_training.

        Default value ``False``.

    random_state : Optional[int]
        Sets a numpy random seed for reproducibility.

        Default value ``None``.

    save_loc: Union[str, bool]
        Destination to save result plot (if provided as a string argument).
        Figure is saved to: `save_loc + 'N_plot-' + estimator_name + "-" + obj + ".pdf"`
        Where `obj` is the objective function of the model prediction and `estimator_name` is the string
        name of the model saved in the WorkFlow object.

        Default value ``False``.


    Returns
    -------
        Dict[str, Tuple[float, pd.DataFrame]]
            Returns a dictionary with each key the string name of the listed objective functions.
            The value of each dictionary is a tuple, the first value is the estimated eta squared value for infinite training data points.
            The second value is a DataFrame with the test/train MSE and their deviations for different dataset slices.
            The DataFrame has the form:

            ========= ============ ============ ===
            -         slice 1 size slice 2 size ...
            test      ...          ...          ...
            train     ...          ...          ...
            test_std  ...          ...          ...
            train_std ...          ...          ...
            ========= ============ ============ ===

            Where test / train are the mean of the MSE scores on the test / train dataset for the models, and test/train_std are their deviations.
    """
    if objective_funcs == None: objective_funcs = list(wf.y.columns)
    rng = np.random.default_rng(seed=random_state)

    #useful local vars
    X,y,std = wf.X, wf.y, wf.std

    # If we have composition groups, we split by unique comps, not by individual datapoints
    if hasattr(wf, 'groups'):
        total_data_count = len(wf.groups.unique())
    else:
        total_data_count = X.shape[0]

    test_count = max(1, test_size if isinstance(test_size, int) else int(test_size*total_data_count))
    N_max = total_data_count - test_count

    result_dict = {}
    for obj in to_list(objective_funcs):
        estimator = clone(wf.get_estimator(estimator_name, obj))#copies hyperparameters but is unfitted, avoids overwriting our prefit model
        print(estimator)
        if isinstance(estimator, PolynomialRegression): X = wf.poly_X

        ## N_min user defined or dimensions of feature set, if poly-lr, we use the number of polynomials as min
        if N_min == None:
            N_min = X.shape[1]
            if isinstance(estimator, PolynomialRegression):
                N_min = len(estimator.polynomials)
            

        results_dict = {}
        for _ in range(repeat):

            # We use different methods if we've defined composition groups
            # composition groups -> split by group
            # otherwise, split datapoints
            if hasattr(wf, 'groups'):
                shuffled_list = rng.choice(list(wf.groups.unique()), N_max, replace=False)
                # A little tricky here, we create a boolean map for each datapoint
                # If datapoint belongs to group in the test set -> True, else false
                test_set = list(wf.groups.loc[
                        [(group_id in shuffled_list[:test_count]) for group_id in wf.groups]
                        ].index)
                rest_set = list(wf.groups.loc[
                        [(group_id not in shuffled_list[:test_count]) for group_id in wf.groups]
                        ].index)
            else:
                shuffled_list = rng.choice(range(N_max), N_max, replace=False)
                test_set = shuffled_list[:test_count]
                rest_set = shuffled_list[test_count:]
            X_test, y_test = X.iloc[test_set], y[obj].iloc[test_set]

            ## Define training_data_count = [N_min,...,N_max] so len(training_data_count) = sample_count  with values which are equidistant on a 1/N scale 
            training_data_count = np.array(1/np.linspace(1/N_min, 1/len(rest_set), sample_count), dtype=int)

            for i, endpoint in enumerate(training_data_count):
                #Initialize dict key if doesn't yet exist
                if not i in results_dict.keys():
                    results_dict[i] = {'data_size': [], 'test': [], 'train': []}
                    
                #Define bin indices and slice out the data
                this_bin = rest_set[:endpoint]
                X_train, y_train = X.iloc[this_bin], y[obj].iloc[this_bin]

                #Train estimator then test
                if isinstance(estimator, AlphaGPR):
                    alpha_train = pd.concat([y[obj].iloc[this_bin], std[obj+"_std"].iloc[this_bin]], axis=1)
                    estimator.fit(X_train, alpha_train)
                else:
                    estimator.fit(X_train, y_train)

                train_error = mean_squared_error(y_train,estimator.predict(X_train))
                test_error = mean_squared_error(y_test,estimator.predict(X_test))
                #Save results
                results_dict[i]['train'].append(train_error)
                results_dict[i]['test'].append(test_error)
                results_dict[i]['data_size'].append(endpoint)
        # Take average score and store the std
        for _, score_dict in results_dict.items():
            score_dict['test_std'] = np.std(score_dict['test'], ddof=0)/np.sqrt(repeat)#TODO check this!
            score_dict['test'] = np.mean(score_dict['test'])
            score_dict['train_std'] = np.std(score_dict['train'], ddof=0)/np.sqrt(repeat)#TODO check this!
            score_dict['train'] = np.mean(score_dict['train'])
            score_dict['data_size'] = np.mean(score_dict['data_size'])
        
        df = pd.DataFrame(results_dict).T.set_index('data_size').T
        lin_fit = LinearRegression().fit(np.array(1/df.columns).reshape(-1, 1), df.loc['train'])
        eta_squared = lin_fit.intercept_
        slope = lin_fit.coef_[0]
        print("Estimated eta**2: {}".format(eta_squared))
        if isinstance(estimator, PolynomialRegression):
            print("P_eff estimated to be {}".format(-slope/eta_squared))
        if plot:
            x_fit = np.linspace(0, 1/df.columns[0], 5)
            plt.plot(figsize=(6,4))
            #plt.title('MSE to 1/N',fontsize=20)
            plt.xlabel('1/N')
            plt.ylabel(r'$\eta^2$')
            # set our y axes to always at least show 1 datapoint from the test score
            plt.ylim([0,max(10*eta_squared, 5*df.loc['test'].min())]) 
            plt.errorbar(1/df.columns,df.loc['test'],yerr=(confidence*df.loc['test_std']),xerr=None,marker='o',label="test", alpha=1,capsize=3, markerfacecolor='white')
            plt.errorbar(1/df.columns,df.loc['train'],yerr=(confidence*df.loc['train_std']),xerr=None,marker='o',label="train", alpha=1,capsize=3, markerfacecolor='white')
            plt.plot(x_fit, slope*x_fit + eta_squared, label="E(Train) fit")
            plt.plot(x_fit, -slope*x_fit + eta_squared, label="E(Predict) expect")
            plt.xticks()
            plt.yticks()
            plt.legend(fontsize=12)
            plt.tight_layout()
            if save_loc: plt.savefig(save_loc + 'N_plot-' + estimator_name + "-" + obj + ".pdf", bbox_inches="tight")
            plt.show()
        result_dict[obj] = (eta_squared, df)
        display(df)
    return result_dict

def performance_plot(wf: WorkFlow, metric: str = "MSE") -> None:
    """
    Generate plot showing time/MAE/MSE/R2 scores of all trained models on training and test data, sorted by `metric`.
    A plot for each objective function in the WorkFlow is automatically generated.

    Parameters
    ----------
    wf: WorkFlow
        WorkFlow object with models to analyze.

    metric: str
        Metric to sort models. Options are:

        ==== === === ==
        time MAE MSE R2
        ==== === === ==

        The models are always sorted by their performance on the test set (rather than training).

        Default value ``"MSE"``

    Returns
    -------
        ``None``
    """
    if not metric in wf.supported_metrics:
        print(" \'{}\' invalid metric, try one of: {}".format(metric, wf.supported_metrics))
        return

    for obj, models in wf.results.items():
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

        sorted_scores = pd.DataFrame(result_dict).sort_values('MSE test', axis=1, ascending=False)
        max_MSE_test = sorted_scores.loc['MSE test'].max()
        max_MSE_train = sorted_scores.loc['MSE train'].max()
        max_MAE_test = sorted_scores.loc['MAE test'].max()
        max_MAE_train = sorted_scores.loc['MAE train'].max()
        max_MSE = max(max_MSE_train, max_MSE_test)
        max_MAE = max(max_MAE_train, max_MAE_test)
        max_time = sorted_scores.loc['time'].max()
        #R2 is already normalized


        labels = sorted_scores.columns

        y = np.arange(len(labels))  # the label locations
        width = 0.1 #the width of the bars

        fig, ax = plt.subplots(figsize=(9, 2*len(labels)), constrained_layout=True)
        error_kw = {'capsize': 1,
                'elinewidth': 0.5}
        rects_time = ax.barh(y + 3.5*width, sorted_scores.loc['time']/max_time, width, xerr=sorted_scores.loc['time_sem']/max_time, label='Time (s)', error_kw=error_kw)
        rects_mse_test = ax.barh(y + 2*width, sorted_scores.loc['MSE test']/max_MSE, width, xerr=sorted_scores.loc['MSE test_sem']/max_MSE, label='MSE test', error_kw=error_kw)
        rects_mae_test = ax.barh(y + width, sorted_scores.loc['MAE test']/max_MAE, width, xerr=sorted_scores.loc['MAE test_sem']/max_MAE, label='MAE test', error_kw=error_kw)
        rects_r2_test = ax.barh(y, sorted_scores.loc['R2 test'], width, xerr=sorted_scores.loc['R2 test_sem'], label='1 - R2 Test', error_kw=error_kw)
        rects_mse_train = ax.barh(y - 1.5*width, sorted_scores.loc['MSE train']/max_MSE, width, xerr=sorted_scores.loc['MSE train_sem']/max_MSE, label='MSE train', error_kw=error_kw)
        rects_mae_train = ax.barh(y - 2.5*width, sorted_scores.loc['MAE train']/max_MAE, width, xerr=sorted_scores.loc['MAE train_sem']/max_MAE, label='MAE train', error_kw=error_kw)
        rects_r2_train = ax.barh(y - 3.5*width, sorted_scores.loc['R2 train'], width, xerr=sorted_scores.loc['R2 train_sem'], label='1 - R2 Train', error_kw=error_kw)

        # Info txt
        ax.set_title('Performance metrics sorted by {}: {}'.format(metric, obj))
        ax.set_xlabel('Score')
        ax.set_ylabel('Model')
        ax.set_xlim(right=1.5)
        ax.set_yticks(y, labels)
        ax.legend(loc='upper right')

        label_fmt = {'label_type': 'edge', 'padding': 5}
        ax.bar_label(rects_time, labels=sorted_scores.loc['time'].map(lambda x: '{:.2e}'.format(x)), **label_fmt)
        ax.bar_label(rects_mse_test, labels=sorted_scores.loc['MSE test'].map(lambda x: '{:.2e}'.format(x)), **label_fmt)
        ax.bar_label(rects_mae_test, labels=sorted_scores.loc['MAE test'].map(lambda x: '{:.2e}'.format(x)), **label_fmt)
        ax.bar_label(rects_r2_test, labels=sorted_scores.loc['R2 test'].map(lambda x: '{:.2e}'.format(x)), **label_fmt)
        ax.bar_label(rects_mse_train, labels=sorted_scores.loc['MSE train'].map(lambda x: '{:.2e}'.format(x)),**label_fmt)
        ax.bar_label(rects_mae_train, labels=sorted_scores.loc['MAE train'].map(lambda x: '{:.2e}'.format(x)),**label_fmt)
        ax.bar_label(rects_r2_train, labels=sorted_scores.loc['R2 train'].map(lambda x: '{:.2e}'.format(x)), **label_fmt)

        fig.tight_layout()

        plt.show()

def create_input(feature_dict: Dict[str, List[float]], steps: int=10, temp: Union[int, float]=-1) -> Tuple[pd.DataFrame, np.array]:
    """
    Create a dataframe grid of input electrolyte compositions from sparse 
    feature vectors, or [min, max] + step values.

    Parameters
    ----------
    feature_dict: Dict[str, list[float]]
        Dictionary with input feature names as keys, and either [min, max] as values,
        or a list of explicit values [val0, val1, val2, ...] to be generated.

    steps: int
        Number of values to generate for each feature axis provided as [min, max] 
        values.


        Default value ``10``

    temp: Union[int, float]
        Temperature to be converted to inverse temperature (1000/(273.15+temp)) and
        added as a feature in the grid. If -1, no temperature is added.

        Default value ``-1``

    Returns
    -------
        ``Tuple[pd.DataFrame, np.array]``

            The DataFrame is the meshgrid spanning all of the feature dimensions given in
            the feature_dict (+ optionally the inverse temperature).

            The array is the set of feature names returned.
    """
    
    inv_temp = 1000/(273.15+temp)
    
    ranges = []
    range_indices = []
    single_values = []
    single_values_indices = []
    i = 0
    for key, val in feature_dict.items():
        if isinstance(val, list) and len(val) == 2:
            min_val, max_val = val
            r = np.linspace(min_val, max_val, steps)
            ranges.append(r)
            range_indices.append(i)
        else:
            if isinstance(val, list):
                val_r = val[0]
            else:
                val_r = val
            single_values.append(val_r)
            single_values_indices.append(i)
        i+=1
        
    grid = np.array(np.meshgrid(*ranges))
    n_data = int(np.product(grid.shape[1:]))
    grid = grid.reshape(len(ranges),n_data)
    keys = np.asarray(list(feature_dict.keys()))
    
    df_range = pd.DataFrame(grid.T, columns=keys[range_indices])
    
    single_values = np.array(single_values)
    
    single_grid = np.ones((len(grid.T), len(single_values)))
    for i in range(len(single_values)):
        single_grid[:,i] = single_grid[:,i]*single_values[i]
    
    df_single = pd.DataFrame(single_grid, columns=keys[single_values_indices]) 
    if temp != -1:
        df_temp = pd.DataFrame(np.ones(len(grid.T))*inv_temp, columns=['inverse temperature'])
        df_combined = pd.concat([df_range, df_single, df_temp], axis=1)
    else:
        df_combined = pd.concat([df_range, df_single], axis=1)
    
    range_keys=keys[range_indices]
    
    return df_combined, range_keys


def predict_conductivity_from_arrhenius_objectives(x_in: pd.DataFrame, wf: WorkFlow,
        model: Union[str, List[str]], beta_0: float, log:bool=False
        ) -> pd.DataFrame:
    """
    Predict the ionic conductivity for given electrolyte compositions at a given temperature.

    Parameters
    ----------
    x\_in: pd.DataFrame
        DataFrame of input feature vectors, matching the DataFrame format for WorkFlow.

    wf: WorkFlow
        LECA WorkFlow object containing trained models for Arrhenius objective functions S0, S1 and S2.

    model: Union[str, List[str]]
        String or list of [string, string, string]. If a single string is provided, the 3 models with that name 
        are drawn from the WorkFlow for predicting S0/S1/S2 respectively. Otherwise, the list position corresponds 
        to the model name selected from the WorkFlow. I.e, the model name passed in model[0] will be used to 
        predict S0, model[1] -> S1 and model[2] -> S2.

    beta_0: float
        beta_0 value used for the Arrhenius surrogate model. This value should have been stored during the transformation
        of measurement data into the Arrhenius surrogate model objectives (See: :func:`.prep.arrhenius`, :func:`.prep.direct_sample_arrhenius`).

    log: bool
        If `log=True` return log_{10}(conductivity). 
        If `log=False` return conductivity. 

        Default value ``False``

    Returns
    -------
        pd.DataFrame
            DataFrame with the following columns structure of input features and conductivity predictions 
            (and their one-sigma uncertainty). x_i here signifies each feature dimension in x\_in.

            =================== === === ==== ============= =============== 
            inverse temperature x_1 x_2 x... conductivity  conducivity_std
            =================== === === ==== ============= =============== 

    """
    
    if isinstance(model,list):
        m1, m2, m3 = model
    else:
        m1 = m2 = m3 = model
    
    x_input = x_in.copy()
    inv_temp = x_input['inverse temperature']
    temp_offset = x_input['inverse temperature'] - beta_0
    x_input.drop('inverse temperature', axis=1, inplace=True)
    pred = wf.predict(x_input, {'S0': m1, 'S1': m2, 'S2': m3}, min_max=True, return_std=True)
    log_cond = (unumpy.uarray(pred['S0'], pred['S0_std'])) \
                - (unumpy.uarray(pred['S1'], pred['S1_std']))*(temp_offset) \
                - (unumpy.uarray(pred['S2'], pred['S2_std']))*(temp_offset)**2
    if log == False:
        cond = unumpy.pow(10,log_cond)
    else:
        cond = log_cond
    cond = pd.DataFrame({'conductivity':unumpy.nominal_values(cond),
                         'conductivity_std':unumpy.std_devs(cond)})
    pred = pd.concat([inv_temp, x_input, cond], axis=1)
    return pred

def predict_conductivity_from_log_conductivity_objective(x_in: pd.DataFrame, wf: WorkFlow, 
        model: str, log:bool=False, objective:str='log conductivity'
        ) -> pd.DataFrame:
    """
    Predict the ionic conductivity for given electrolyte compositions at a given temperature. This function
    is to be used with single objective workflows.

    Parameters
    ----------
    x\_in: pd.DataFrame
        DataFrame of input feature vectors, matching the DataFrame format for WorkFlow.

    wf: WorkFlow
        LECA WorkFlow object containing trained models for the single objective
        function (generally "log conductivity").

    model: str
        String name of model to use for prediction.

    log: bool
        If `log=True` return log_{10}(conductivity). 
        If `log=False` return conductivity. 

        Default value ``False``

    objective: str
        String name of the objective function for the trained models in the WorkFlow.

        Default value ``"log conductivity"``

    Returns
    -------
        pd.DataFrame
            DataFrame with the following columns structure of input features and conductivity predictions 
            (and their one-sigma uncertainty). x_i here signifies each feature dimension in x\_in.

            =================== === === ==== ============= =============== 
            inverse temperature x_1 x_2 x... conductivity  conducivity_std
            =================== === === ==== ============= =============== 

    """
    
    x_input = x_in.copy()
    inv_temp = x_input['inverse temperature']
    pred = wf.predict(x_input, {objective: model}, min_max=True, return_std=True)
    log_cond = unumpy.uarray(pred[objective], pred[objective+'_std'])
    if log == False:
        cond = unumpy.pow(10,log_cond)
    else:
        cond = log_cond
    cond = pd.DataFrame({'conductivity':unumpy.nominal_values(cond),
                         'conductivity_std':unumpy.std_devs(cond)})
    pred = pd.concat([inv_temp, x_input, cond], axis=1)
    return pred

def plot_1D(wfs: List[WorkFlow], models: List[str], feature_dict:Dict[str, List[float]], beta_0_list:List[float], 
        temperatures:Union[int, float, List[int], List[float]]=20, steps:int=50, ylim:Optional[Tuple[float,float]]=None, 
        multiply_by_salt:bool=False, log:bool=False, 
        model_labels:Optional[List[str]]=None, wf_labels:Optional[List[str]]=None, 
        confidence:float = 1.0, save_loc: Union[str, bool] = False, objective:str='log conductivity',
        indicate_max:Tuple[Optional[str],Union[int, float],Union[int,float]]=(None,0.8, -1)) -> None:
    """
        1-dimensional slice along one feature for models predicted conductivity / log(conductivity). 
        The function can be used for models trained directly on ionic conductivity or trained on the Arrhenius objectives.
        A single plot will be rendered which shows the predictions for the argument defined WorkFlow(s) for every model 
        at every temperature for the given feature ranges/values.

    Parameters
    ----------
    wfs: List[WorkFlow]
        List of LECA WorkFlow object(s) containing trained models for the single objective
        function (generally "log conductivity"). 
        Each WorkFlow in the wfs list should correspond to a WorkFlow label and beta_0 
        value in the wf_labels list and beta_0_list arguments, respectively.

    models: List[str] 
        List of string names of model(s) to use for prediction. 

    feature_dict: Dict[str, List[float]] 
        Dictionary with input feature names as keys, and either [min, max] as values,
        or a list of explicit values [val0, val1, val2, ...] to be used to generate
        predictions.

    beta_0_list: List[float] 
        List of the beta_0 values for the Arrhenius fits for each trained WorkFlow to be plotted
        (See :func:`.prep.arrhenius` or :func:`.prep.direct_sample_arrhenius`). **If the value
        ``-1`` is passed in the list, this signifies that the WorkFlow doesn't use the Arrhenius
        surrogate model and the predictions are not back-transformed.**

    temperatures: Union[int, float, List[int], List[float]]
        Set of temperature(s) to use for plotting predictions.

        Default value ``20``

    steps: int
        Number of steps for generating values between feature_dict[feature]: min to max.

        Default value ``50``

    ylim: Optional[Tuple[float,float]]
        Optional parameter to set fixed boundaries for the y-axis of the plot.

        Default value ``None``

    multiply_by_salt: bool
        Whether to multiply the prediction by the salt content (feature: "x_LiSalt"). This
        is necessary to back-transform to conductivity if the models were trained with the
        objective function conductivity/x_Lisalt.

        Default value ``False``

    log: bool
        If `log=True` return log_{10}(conductivity). 
        If `log=False` return conductivity. 

        Default value ``False``

    model_labels: Optional[List[str]]
        String values for labeling the models in the plots. Can be useful for abbreviating
        otherwise verbose string model names. If ``None``, by default uses the model names
        given in the `models` argument.

        Default value ``None``

    wf_labels: Optional[List[str]]
        Optionally append a label for different WorkFlows to the plotted predictions. If ``None``
        nothing is added. Note: If a list is passed, it should correspond 1:1 with the list of WorkFlows
        passed to the `wfs` argument.

        Default value ``None``

    confidence: float
        Scalar value to multiply the estimated uncertainty. By default this value is ``1.0`` which results in the
        plotted errorbars showing one standard-deviation. E.g. ``confidence=1.96`` would then reflect an
        approximate 95\% confidence interval.

        Default value ``1.0``

    save_loc: Union[str, bool]
        Boolean or string to indicate whether and where to to save the plot. If ``False`` no plot is saved, otherwise:
        Depending on other passed arguments, the naming scheme follows:

        multiply_by_salt==True and indicate_max[0] != None :: save_loc+'slice_1D_{varied_feature}_multiply_by_salt_indicate.pdf

        multiply_by_salt==True and indicate_max[0] == None :: plt.savefig(save_loc+'slice_1D_{varied_feature}_multiply_by_salt.pdf
        
        multiply_by_salt==False and indicate_max[0] != None :: plt.savefig(save_loc+'slice_1D_{varied_feature}_indicate.pdf

        multiply_by_salt==False and indicate_max[0] == None :: plt.savefig(save_loc+'slice_1D_{varied_feature}.pdf

        Default value ``False``

    objective: str
        String name of the objective function for the trained models in the WorkFlow.

        Default value ``'log conductivity'``

    indicate_max: Tuple[Optional[str],Union[int, float],Union[int,float]]
        indicate_max[0] : ``None`` or String. If None do nothing, if String, plot a vertical dashed line 
        indicating range of varied_feature within which the predicted objective is greater than indicate_max[1]*global_objectve_max
        as predicted by the model string name passed to indicate_max[0]. E.g. indicate_max[1]=0.8 plots the top 20th percentile
        objective value prediction range.

        indicate_max[2] : If ``-1`` set the vertical dashed line limit as the minimum predicted objective value,
        otherwise, optionally set the objective value to which the vertical dashed lines are plotted. 
        E.g. indicate_max[2]=0 will plot the vertical dashed lines to from the predicted objective value down to y=0.

        Default value ``(None,0.8, -1))``

    Returns
    -------
        ``None``
    """
    
    if isinstance(temperatures, int) or isinstance(temperatures, float):
    
        temperatures = [temperatures]
    
    if model_labels == None:
        print("No labels given.")
        model_labels = [str(model) for model in models]
    
    if wf_labels == None:
        print("No labels given.")
        wf_labels = ['' for model in models]
        
    fig, ax = plt.subplots(figsize=(8,6))
    
    min_v = None
    
    for temp in temperatures:
        x_input, range_key = create_input(feature_dict,steps,temp)
        min_cond = 10
        for model, model_label in zip(models, model_labels):
            for wf, wf_label, beta_0 in zip(wfs, wf_labels, beta_0_list):
                if beta_0 == -1:
                    specific_prediction = predict_conductivity_from_log_conductivity_objective(x_input, wf, model, log, objective)
                else:
                    specific_prediction = predict_conductivity_from_arrhenius_objectives(x_input, wf, model, beta_0, log)

                if multiply_by_salt == True:
                    if log == False:
                        specific_prediction['conductivity'] = specific_prediction['conductivity']*x_input['x_LiSalt']
                        specific_prediction['conductivity_std'] = specific_prediction['conductivity_std']*x_input['x_LiSalt']
                    else:
                        specific_prediction['conductivity'] = specific_prediction['conductivity']+np.log10(x_input['x_LiSalt'])
                    
                if min(specific_prediction['conductivity']) < min_cond:
                    min_cond = min(specific_prediction['conductivity'])
                
                if model == indicate_max[0]:
                    max_pred = max(specific_prediction['conductivity'])
                    larger_cond = []
                    larger_value = []
                    for i in range(len(specific_prediction['conductivity'])):
                        if specific_prediction['conductivity'].iloc[i]>indicate_max[1]*max_pred:
                            larger_cond.append(specific_prediction['conductivity'].iloc[i])
                            larger_value.append(specific_prediction[range_key].iloc[i])
                    larger_cond = np.asarray(larger_cond)
                    larger_value = np.asarray(larger_value)
                    min_v =  np.where(larger_value == min(larger_value))[0]
                    max_v = np.where(larger_value == max(larger_value))[0] 
                    
                ax.plot(specific_prediction[range_key[0]], specific_prediction['conductivity'], label=wf_label+model_label+" {} °C".format(temp))
                min_c = specific_prediction['conductivity'] - specific_prediction['conductivity_std']*confidence
                max_c = specific_prediction['conductivity'] + specific_prediction['conductivity_std']*confidence
                ax.fill_between(specific_prediction[range_key[0]], min_c, max_c, alpha=0.2)
        
        if indicate_max[2] != -1:
            min_cond = indicate_max[2]
        if min_v != None:
            last_plot = fig.gca().lines[-1]
            previous_color = last_plot.get_color()
            ax.vlines(larger_value[min_v], ymin=min_cond, ymax  = larger_cond[min_v], color=previous_color, linestyles='dashed')
            ax.vlines(larger_value[max_v], ymin=min_cond, ymax  = larger_cond[max_v], color=previous_color, linestyles='dashed')
            print(larger_value[min_v])
            print(larger_value[max_v])
            print()
                
    if ylim == None:
        pass
    else:
        ax.set_ylim(ylim)

    ax.legend(loc='upper right')
    ax.set_xlabel('$'+range_key[0].replace('_','_\\mathrm{')+'}$')
    if multiply_by_salt == True:
        if log==False:
            ax.set_ylabel('$\sigma$ [S/cm]')
        else: 
            ax.set_ylabel('$\log(\sigma)$')
    else:
        if log==False:
            ax.set_ylabel('$\sigma/x_\\mathrm{LiSalt}$ [S/cm]')
        else:
            ax.set_ylabel('$\log(\sigma/x_\\mathrm{LiSalt})$')
            
    if log==True:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))    
        
    if save_loc: 
        if multiply_by_salt==True:
            if indicate_max[0] != None:
                plt.savefig(save_loc+'slice_1D_{}_multiply_by_salt_indicate.pdf'.format(range_key[0]).replace(":","_"), bbox_inches="tight")
            else:
                plt.savefig(save_loc+'slice_1D_{}_multiply_by_salt.pdf'.format(range_key[0]).replace(":","_"), bbox_inches="tight")
        else:
            if indicate_max[0] != None:
                plt.savefig(save_loc+'slice_1D_{}_indicate.pdf'.format(range_key[0]).replace(":","_"), bbox_inches="tight")
            else:
                plt.savefig(save_loc+'slice_1D_{}.pdf'.format(range_key[0]).replace(":","_"), bbox_inches="tight")
    plt.show()
            
def plot_1D_Sx(wfs: List[WorkFlow], models: List[str], feature_dict:Dict[str, List[float]],
        beta_0_list:List[float], steps:int=50, objectives:List[str]=['S0', 'S1', 'S2'], 
        ylim:Optional[Tuple[float,float]]=None, multiply_by_salt:bool=False, 
        model_labels:Optional[List[str]]=None, wf_labels:Optional[List[str]]=None, 
        confidence:float = 1.0, save_loc: Union[str, bool] = False) -> None:
    """
        1-dimensional slice along one feature for models predicted arrhenius objectives S0, S1 and S2.
        Three plots will be rendered which show the S0, S1 and S2 predictions for the argument defined
        WorkFlow(s) for every model for the given feature ranges/values.

    Parameters
    ----------
    wfs: List[WorkFlow]
        List of LECA WorkFlow object(s) containing trained models for the three Arrhenius model
        coefficients (typically S0, S1 and S2, though these names can be modified with 
        the `objectives` argument).
        Each WorkFlow in the wfs list should correspond to a WorkFlow label and beta_0 
        value in the wf_labels list and beta_0_list arguments, respectively.

    models: List[str] 
        List of string names of model(s) to use for prediction. 

    feature_dict: Dict[str, List[float]] 
        Dictionary with input feature names as keys, and either [min, max] as values,
        or a list of explicit values [val0, val1, val2, ...] to be used to generate
        predictions.

    beta_0_list: List[float] 
        List of the beta_0 values for the Arrhenius fits for each trained WorkFlow to be plotted
        (See :func:`.prep.arrhenius` or :func:`.prep.direct_sample_arrhenius`).

    steps: int
        Number of steps for generating values between feature_dict[feature]: min to max.

        Default value ``50``

    ylim: Optional[Tuple[float,float]]
        Optional parameter to set fixed boundaries for the y-axis of the plots.

        Default value ``None``

    multiply_by_salt: bool
        Whether to multiply the prediction by the salt content (feature: "x_LiSalt"). This
        is necessary if the models were trained with the objective function log(conductivity/x_Lisalt).
        If ``True`` S0 predictions are transformed: S0_output = S0_pred + log10(x_LiSalt)

        Default value ``False``

    model_labels: Optional[List[str]]
        String values for labeling the models in the plots. Can be useful for abbreviating
        otherwise verbose string model names. If ``None``, by default uses the model names
        given in the `models` argument.

        Default value ``None``

    wf_labels: Optional[List[str]]
        Optionally append a label for different WorkFlows to the plotted predictions. If ``None``
        nothing is added. Note: If a list is passed, it should correspond 1:1 with the list of WorkFlows
        passed to the `wfs` argument.

        Default value ``None``

    confidence: float
        Scalar value to multiply the estimated uncertainty. By default this value is ``1.0`` which results in the
        plotted errorbars showing one standard-deviation. E.g. ``confidence=1.96`` would then reflect an
        approximate 95\% confidence interval.

        Default value ``1.0``

    save_loc: Union[str, bool]
        Boolean or string to indicate whether and where to to save the plot. If ``False`` no plot is saved, otherwise:
        The naming scheme follows: save_loc+'slice_1D_Sx_{varied_feature}.pdf

        Default value ``False``

    Returns
    -------
        ``None``
    
    """
    
    if model_labels == None:
        print("No labels given.")
        model_labels = [str(model) for model in models]
    
    if wf_labels == None:
        print("No labels given.")
        wf_labels = ['' for model in models]
        
    
    x_input, range_key = create_input(feature_dict,steps,-1)
    len_o = len(objectives)
    fig, ax = plt.subplots(nrows=1, ncols=len_o, figsize=(8*len_o,6))
    
    for model, model_label in zip(models, model_labels):
        if isinstance(model,list):
            m1, m2, m3 = model
        else:
            m1 = m2 = m3 = model
        for wf, wf_label, beta_0 in zip(wfs, wf_labels, beta_0_list):
            pred = wf.predict(x_input, {'S0': m1, 'S1': m2, 'S2': m3}, min_max=True, return_std=True)
            if multiply_by_salt == True:
                pred['S0'] = pred['S0'] + np.log10(x_input['x_LiSalt'])
            specific_prediction = pd.concat([x_input, pred], axis=1)
            for i in range(len_o): 
                obj = objectives[i]
                ax[i].plot(specific_prediction[range_key[0]], specific_prediction[obj], label=model_label+wf_label)
                min_c = specific_prediction[obj] - specific_prediction[obj+'_std']*confidence
                max_c = specific_prediction[obj] + specific_prediction[obj+'_std']*confidence
                ax[i].fill_between(specific_prediction[range_key[0]], min_c, max_c, alpha=0.2)
    if ylim == None:
        pass
    else:
        ax[i].set_ylim(ylim)
    
    for i in range(len_o):
        ax[i].legend(loc='best')
        ax[i].set_xlabel('$'+range_key[0].replace('_','_\\mathrm{')+'}$')
        ax[i].set_ylabel(objectives[i])
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    if save_loc: plt.savefig(save_loc+'slice_1D_Sx_{}.pdf'.format(range_key[0]).replace(":","_"), bbox_inches="tight")
    plt.show()


def plot_2D(wf: WorkFlow, model: Union[str, List[str]], feature_dict:Dict[str, List[float]], 
        temp:Union[int,float], beta_0:Union[int,float], steps:int=50, 
        restriction: List[str] =['x_EC', 'x_EMC', 'x_LiSalt'], 
        multiply_by_salt:bool=False, log:bool=False, focus:Union[bool,pd.DataFrame]=False, 
        save_loc: Union[str, bool] = False, objective:str='log conductivity', **kwargs
        ) -> None:
    """
        2-dimensional slice along two features for predicted conductivity / log(conductivity). 
        The function can be used for model(s) trained directly on ionic conductivity or trained on the Arrhenius objectives.
        A single plot will be rendered which shows the predictions for the given feature ranges/values.

    Parameters
    ----------
    wf: WorkFlow
        LECA WorkFlow object containing the trained model(s) for predicting the
        objective function (generally "log conductivity"). 

    model: Union[str, List[str]]
        String or list of string names of model(s) to use for prediction. If a single string
        is passed, the same model is used for all objectives. If the WorkFlow is trained
        on Arrhenius objectives they are then back-transformed.

    feature_dict: Dict[str, List[float]] 
        Dictionary with input feature names as keys, and either [min, max] as values,
        or a list of explicit values [val0, val1, val2, ...] to be used to generate
        predictions.

    temp: Union[int, float]
        Temperature to use for plotting predictions.

    beta_0: Union[int, float] 
        beta_0 value for the Arrhenius fits for the WorkFlow to be plotted
        (See :func:`.prep.arrhenius` or :func:`.prep.direct_sample_arrhenius`). **If the value
        ``-1`` is passed, this signifies that the WorkFlow doesn't use the Arrhenius
        surrogate model and the predictions are not back-transformed.**

    steps: int
        Number of steps for generating values between feature_dict[feature]: min to max.

        Default value ``50``

    restriction: List[str] 
        Set limited feature values. If the sum of the input features declared in this list
        is greater than 1, the prediction is excluded. This argument can be used to set
        boundaries for impossible electrolyte compositions.

        Default value ``['x_EC', 'x_EMC', 'x_LiSalt']``

    multiply_by_salt: bool
        Whether to multiply the prediction by the salt content (feature: "x_LiSalt"). This
        is necessary to back-transform to conductivity if the models were trained with the
        objective function conductivity/x_Lisalt.

        Default value ``False``

    log: bool
        If `log=True` return log_{10}(conductivity). 
        If `log=False` return conductivity. 

        Default value ``False``
    
    focus: Union[bool, pd.DataFrame]
        If False, do nothing. If pd.DataFrame (1-row DataFrame with input features columns)
        a black circle will be plotted on the 2D plots corresponding to the focus point.

        Default value ``False``

    save_loc: Union[str, bool]
        Boolean or string to indicate whether and where to to save the plot. If ``False`` no plot is saved, otherwise:
        Depending on other passed arguments, the naming scheme follows: 
        save_loc+'slice_2D_{varied_features}.pdf

        Default value ``False``

    objective: str
        String name of the objective function for the trained models in the WorkFlow.

        Default value ``'log conductivity'``

    **kwargs:
        Keyword arguments passed to matplotlib.pyplot.countourf.


    Returns
    -------
        ``None``
    """
            
    x_input, range_keys = create_input(feature_dict,steps,temp)
    
    fig, ax = plt.subplots(figsize=(6,5))
    fig.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    title = ''
    for index in x_input.columns:
        if index not in range_keys and index != 'inverse temperature':
            title = title + '{}={} '.format('$'+index.replace('_','_\\mathrm{')+'}$', round(feature_dict[index],3))
    
    if beta_0 == -1:
        specific_prediction = predict_conductivity_from_log_conductivity_objective(x_input, wf, model, log, objective)
    else:
        specific_prediction = predict_conductivity_from_arrhenius_objectives(x_input, wf, model, beta_0, log)
    if multiply_by_salt == True:
        if log == False:
            specific_prediction['conductivity'] = specific_prediction['conductivity']*x_input['x_LiSalt']
            specific_prediction['conductivity_std'] = specific_prediction['conductivity_std']*x_input['x_LiSalt']
        else:
            specific_prediction['conductivity'] = specific_prediction['conductivity']+np.log10(x_input['x_LiSalt'])

    #apply restrictions
    applied_restriction = np.sum(specific_prediction[restriction], axis=1)
    index = np.where(applied_restriction>1)
    index_conductivity = np.where(specific_prediction.columns == 'conductivity')[0]
    
    for i in index:
        specific_prediction.iloc[i,index_conductivity] = 0
    
    data = np.array(specific_prediction[range_keys])
    conductivity = np.array(specific_prediction['conductivity'])
    

    cont = ax.contourf(data[0:steps:1,0], data[0:-1:steps,1], conductivity.reshape(steps,steps), cmap='inferno', **kwargs)
    #if focus == True:
    #    max_cond = np.where(specific_prediction['conductivity'] == max(specific_prediction['conductivity']))[0]
    #    ax.scatter(data[max_cond,0], data[max_cond,1], marker='o', edgecolors='black',s=150, facecolors='none', linewidth=2)
    ax.set_xlabel('$'+range_keys[0].replace('_','_\\mathrm{')+'}$')
    ax.set_ylabel('$'+range_keys[1].replace('_','_\\mathrm{')+'}$')
    ax.set_title(title)
    if multiply_by_salt == True:
        if log==False:
            fig.colorbar(cont, cax=cax, label='$ \sigma$ [S/cm]', format="%1.3f")
        else:
            fig.colorbar(cont, cax=cax,label='$\log(\sigma)$', format="%1.3f")
    else:
        if log==False:
            fig.colorbar(cont, cax=cax,label='$ \sigma/x_\\mathrm{LiSalt}$ [S/cm]', format="%1.3f")
        else:
            fig.colorbar(cont, cax=cax,label='$\log(\sigma/x_\\mathrm{LiSalt}$)', format="%1.3f")
    
    if isinstance(focus, pd.DataFrame):
        ax.scatter(focus[range_keys[0]],focus[range_keys[1]], 150, marker='o', facecolors='none', edgecolors='black', linewidth=2)
    
    if save_loc: plt.savefig(save_loc+'slice_2D_{}.pdf'.format(range_keys).replace(":","_"), bbox_inches="tight")
    plt.show()

def plot_2D_Sx(wf: WorkFlow, model: Union[str, List[str]], feature_dict:Dict[str, List[float]], 
        steps:int=50, restriction: List[str] =['x_EC', 'x_EMC', 'x_LiSalt'], 
        multiply_by_salt:bool=False, focus:Union[bool,pd.DataFrame]=False, 
        save_loc: Union[str, bool] = False, objectives:List[str]=['S0', 'S1', 'S2'], 
        **kwargs) -> None:
    """
        2-dimensional slice along two features for predicted Arrhenius objective values (typically S0, S1 and S2).
        Three plots will be rendered which show the coefficient predictions for the given feature ranges/values.

    Parameters
    ----------
    wf: WorkFlow
        LECA WorkFlow object containing the trained models for predicting the Arrhenius
        objective functions. 

    model: Union[str, List[str]]
        String or list of string names of model(s) to use for prediction. If a single string
        is passed, the same model is used for all objectives.

    feature_dict: Dict[str, List[float]] 
        Dictionary with input feature names as keys, and either [min, max] as values,
        or a list of explicit values [val0, val1, val2, ...] to be used to generate
        predictions.

    steps: int
        Number of steps for generating values between feature_dict[feature]: min to max.

        Default value ``50``

    restriction: List[str] 
        Set limited feature values. If the sum of the input features declared in this list
        is greater than 1, the prediction is excluded. This argument can be used to set
        boundaries for impossible electrolyte compositions.

        Default value ``['x_EC', 'x_EMC', 'x_LiSalt']``

    multiply_by_salt: bool
        Whether to multiply the prediction by the salt content (feature: "x_LiSalt"). This
        is necessary if the models were trained with the objective function log(conductivity/x_Lisalt).
        If ``True`` S0 predictions are transformed: S0_output = S0_pred + log10(x_LiSalt)

        Default value ``False``

    focus: Union[bool, pd.DataFrame]
        If False, do nothing. If pd.DataFrame (1-row DataFrame with input features columns)
        a black circle will be plotted on the 2D plots corresponding to the focus point.

        Default value ``False``

    save_loc: Union[str, bool]
        Boolean or string to indicate whether and where to to save the plot. If ``False`` no plot is saved, otherwise:
        Depending on other passed arguments, the naming scheme follows: 
        save_loc+'slice_2D_Sx_{varied_features}.pdf

        Default value ``False``

    objectives: str
        String name of the Arrhenius coefficients for the trained models in the WorkFlow.

        Default value ``['S0', 'S1', 'S2']``

    **kwargs:
        Keyword arguments passed to matplotlib.pyplot.countourf.


    Returns
    -------
        ``None``
    """
            
    x_input, range_keys = create_input(feature_dict,steps)
    len_o = len(objectives)
    fig, ax = plt.subplots(nrows=1, ncols=len_o, figsize=(8*len_o,6))
    divider = [make_axes_locatable(axis) for axis in ax]
    cax = [div.append_axes('right', size='5%', pad=0.05) for div in divider]
    
    title = ''
    for index in x_input.columns:
        if index not in range_keys and index != 'inverse temperature':
            title = title + '{}={} '.format('$'+index.replace('_','_\\mathrm{')+'}$', round(feature_dict[index],3))

    if isinstance(model,list):
        m1, m2, m3 = model
    else:
        m1 = m2 = m3 = model
    
    # Build prediction dataframe
    pred = wf.predict(x_input, {'S0': m1, 'S1': m2, 'S2': m3}, min_max=True, return_std=True)
    if multiply_by_salt == True:
        pred['S0'] = pred['S0'] + np.log10(x_input['x_LiSalt'])
    specific_prediction = pd.concat([x_input, pred], axis=1)

    #apply restrictions
    applied_restriction = np.sum(specific_prediction[restriction], axis=1)
    specific_prediction.loc[np.where(applied_restriction>1)][objectives] = np.nan
    
    data = np.array(specific_prediction[range_keys])
    for i in range(len_o):
        Si = np.array(specific_prediction[objectives[i]])

        cont = ax[i].contourf(data[0:steps:1,0], data[0:-1:steps,1], Si.reshape(steps,steps), cmap='inferno', **kwargs)
        ax[i].set_xlabel('$'+range_keys[0].replace('_','_\\mathrm{')+'}$')
        ax[i].set_ylabel('$'+range_keys[1].replace('_','_\\mathrm{')+'}$')
        ax[i].set_title(title)
        fig.colorbar(cont, cax=cax[i], label=objectives[i], format="%1.3f")
    
        if isinstance(focus, pd.DataFrame):
            ax[i].scatter(focus[range_keys[0]],focus[range_keys[1]], 150, marker='o', facecolors='none', edgecolors='black', linewidth=2)
        
    fig.tight_layout(pad=1)
    if save_loc: plt.savefig(save_loc+'slice_2D_Sx_{}_{}.pdf'.format(range_keys[0],range_keys[1]).replace(":","_"), bbox_inches="tight")
    plt.show()

def predict_arrhenius_fit(x: pd.DataFrame, beta_0: float) -> Tuple[np.array, unumpy.uarray]:
    '''
        Predict the arrhenius fit from a pandas Series for a single data point.
        
        Parameters
        ----------
            x : pd.DataFrame
                Containing values for the arrhenius objective functions and their standard deviations.

            beta_0 : float
                beta_0 temperature of the Arrhenius fit.
        
        Returns
        -------
            ``None``
    '''
    T = np.linspace(243, 343, 100)
    inv_T = 1000/T
    S0 = unumpy.uarray(x['S0'], x['S0_std'])
    S1 = unumpy.uarray(x['S1'], x['S1_std'])
    S2 = unumpy.uarray(x['S2'], x['S2_std'])
    
    log_cond = S0 - S1*(inv_T-beta_0) - S2*(inv_T-beta_0)**2
    return inv_T, log_cond

def visualize_arrhenius_fit(
        features:List[str],
        beta_0:float,
        x_arrhenius:pd.DataFrame,
        indices:List[int]=[],
        labels:Optional[List[str]]=None,
        true_x_dfs: Optional[pd.DataFrame]=None,
        plot_std:bool=True,
        true_objective:str='log(conductivity/x_LiSalt)',
        colors: Optional[List[str]] = None,
        y_label: Optional[str] = None,
        individual_data_df:Optional[pd.DataFrame]=None,
        confidence:float = 1.0,
        save_loc: Union[str, bool] = False,
        save_idx: Union[int, float, str]=0,
        title: Optional[str] = None
        ) -> None:
    '''
        Can be used to visualize arrhenius fit or predicted arrhenius fits.
        
        Parameters
        ----------
            features : List[str]
                List with all relevant (arrhenius) features
                
            beta_0 : float
                beta_0 value of Arrhenius fit
                
            x_arrhenius : pd.DataFrame
                DataFrame containing the results of the Arrhenius fit.
            
            indices : List[int]
                List containing the indices of each Arrhenius fit in the ``x_arrhenius`` DataFrame that should be plotted.
                
                Default value ``[]``.
                
            labels : Optional[List[str]]
                Additional labels to assign to each Arrhenius fit. If None is provided no labels are used.
                
                Default value ``None``
            
            true_x_dfs : Optional[pd.DataFrame]
                DataFrame containing the true mean measured values for the ionic conductivity. If ``None`` is provided 
                no experimental data is plotted.
                
                Default value ``None``
                
            plot_std : Optional[bool]
                Whether to plot the standard deviations of Arrhenius fits.
                
                Default value ``True``
                
            true_objective : str
                Name of the column in the ``true_x`` DataFrame that contains the measured (and transformed) ionic conductivity.
                
                Default value ``log(conductivity/x_LiSalt)``
                
            colors : Optional[List[str]]
                List of colors for plotting data from different true_x_dfs. If None is provided, black is used as a default color for all data.
                
                Default value ``None``

            y_label : Optional[str]
                Custom label for y axis. If None is given, true_objective will be used.
                
                Default value ``None``
               
            individual_data : Optional[pd.DataFrame]
                DataFrame containing the true individual measured values for the ionic conductivity. If ``None`` is provided 
                no individual experimental data is plotted.
                
                Default value ``None``

            confidence: float
                Scalar value to multiply the estimated uncertainty. By default this value is ``1.0`` which results in the
                plotted errorbars showing one standard-deviation. E.g. ``confidence=1.96`` would then reflect an
                approximate 95\% confidence interval.

                Default value ``1.0``
               
            save_loc : Union[str, bool]
                Destination to save result plot (if provided as a string argument).
                Figure is saved to: save_loc+'arrhenius_fit_{}.pdf'.format(save_idx)`
                Where `obj` is the objective function of the model prediction.
            
                Default value ``False``
                
            save_idx: Union[int, float, str]
                Additional index to save the plot.

                Default value ``0``

            title: Optional[str] = None
                Optionally set a title for the plot. If ``None``, exclude title.
                
                Default value ``None``
        
        Returns
        -------
            ``None``
        
    '''
    true_label_mean = 'Mean Exp. data'
    true_label = 'Exp. data'
    
    if y_label == None:
        y_label = true_objective
    
    
    if labels == None:
        print("No labels given.")
        labels = ['' for i in range(len(indices))]
        
    if colors == None:
        colors = ['black' for i in range(len(true_x_dfs))]
        
    fig, ax = plt.subplots(figsize=(6,4))
    
    for x_arrh, index, label in zip(x_arrhenius, indices, labels):
        
        formulation = x_arrh.loc[index]
        i=0
        for true_x in true_x_dfs:
            if isinstance(true_x, pd.DataFrame) or isinstance(true_x, pd.Series):
                #exp_indices = [i if np.array_equal(true_x[features].iloc[i], formulation[features]) is True else 0 for i in range(len(true_x[features]))]
                exp_indices = np.where(np.array(true_x[features].sum(axis=1)) == formulation[features].sum())
                #exp_indices = np.where(np.array(exp_indices) != 0)
                ax.errorbar(true_x['inverse temperature'].iloc[exp_indices], 
                             true_x[true_objective].iloc[exp_indices],
                             yerr=true_x[true_objective+'_std'].iloc[exp_indices]*confidence, 
                             fmt='o', color=colors[i], markerfacecolor='white', 
                             label=true_label_mean)
                true_label_mean=''
                i+=1
        
        if isinstance(individual_data_df, pd.DataFrame) or isinstance(individual_data_df, pd.Series):
            exp_indices = np.where(np.array(individual_data_df[features].sum(axis=1)) == formulation[features].sum())
            ax.plot(individual_data_df['inverse temperature'].iloc[exp_indices], 
                             individual_data_df[true_objective].iloc[exp_indices], 'rx', label=true_label)
            true_label=''

        T, conductivity = predict_arrhenius_fit(formulation, beta_0)

        ax.plot(T, unumpy.nominal_values(conductivity), '-', label=label)
        if plot_std==True:
            min_c = unumpy.nominal_values(conductivity)-unumpy.std_devs(conductivity)*confidence
            max_c = unumpy.nominal_values(conductivity)+unumpy.std_devs(conductivity)*confidence
            last_plot = fig.gca().lines[-1]
            previous_color = last_plot.get_color()
            ax.fill_between(T, min_c, max_c, color=previous_color, alpha=0.2)
            
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xmin = xmin*1.03
    ymin = ymin/1.03
    ax.text(xmin, ymin, s=str(x_arrhenius[0].loc[index]), fontsize=8)
    
    if title != None:
        ax.set_title(title)

    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('1000 / $T$ [1/K]')
    ax.set_ylabel(y_label)
    
    if save_loc: plt.savefig(save_loc+'arrhenius_fit_{}.pdf'.format(save_idx), bbox_inches="tight")
    plt.show()
    
def extract_results(wf:WorkFlow) -> pd.DataFrame:
    '''
        Extract the results stored in a workflow.
        
        Parameters
        ----------
        wf : WorkFlow
            A workflow in which models are already trained.
            
        Returns
        -------
            pd.DataFrame
                Data frame containing all results of a workflow training.
                ============= ============= ===
                Model name    timings       ...
                ============= ============= ===

    '''
    results = []
    for obj, models in wf.results.items():
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

        obj_results = pd.DataFrame(result_dict).transpose()
        results.append(obj_results)
    return results
