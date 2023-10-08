import os
import json
import csv
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import hdbscan
from sklearn import preprocessing
from sklearn.feature_selection import RFECV #Recursive feature elimination using cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
# For type annotations
from typing import List, Tuple, Union, Optional, Callable


def import_data(
        json_directories: List[str] = [],
        path_to_csv: str = 'json_import.csv',
        mode: str = 'append',
        verbose: bool = True) -> pd.DataFrame:
    """Function for importing json data following the BIG-MAP json formatting standard.

    Parameters
    ----------

    json_directories : List[str], optional
        String list of the directories (relative to notebook directory) to pull the json files from.
        It will pull every .json file in each listed directory and load them into a labeled DataFrame

        By default ``[]``.

    path_to_csv : str, optional
        Path to formatted csv file. If file already exists, import_data will extend the file

        By default "json_import.csv".

    mode : str, optional
        Import mode.
        Choose among:
        
        - "read_csv", read in the data from the csv file (ignores the data folders)
        - "append", read in data from csv file, append all data from the listed jsons files and save to csv
        - "overwrite", read in data from jsons, replace csv file with data from jsons

        By default "append".

    verbose : bool
        Toggle whether to output information about import process

        By default ``True``

    Returns
    -------
    ``DataFrame``
        Pandas DataFrame with labeled columns and one row for each measurement

        If the data in the json file cannot be properly parsed, returns instead the DataFrame for the failed .json import to trace the issue
    """
    
    # Append .csv if necessary
    path_to_csv = path_to_csv if path_to_csv.endswith('.csv') else path_to_csv + '.csv'
    
    global_df = pd.read_csv(path_to_csv, float_precision='round_trip') if (os.path.exists(path_to_csv) and mode != 'overwrite') else pd.DataFrame()#float_precision important here, otherwise reading csv rounds to lower precision and breaks dup. checks

    if mode == 'read_csv': 
        print("Import Summary:\n\tLabels: {}\n\tData points: {}".format(list(global_df.columns), global_df.shape[0]))
        return global_df #Simply read the csv file and return data

    duplicate_count = 0

    for path_to_json in json_directories:
        try:
            json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        except:
            print("Warning file \'{}\' not found".format(path_to_json))
            continue

        for filename in json_files:
            with open(path_to_json + filename, 'r', encoding= 'unicode_escape') as json_file:
                data = json.load(json_file)
                df = pd.DataFrame(data)

            #Had some issues with capitalization of columns/indexes -- casting them all to lowercase for consistency
            df.columns = map(str.lower, df.columns)
            df.index = map(str.lower, df.index)

            try:
                data_dict = {
                    'conductivity' : df["conductivity experiment"]["conductivity data"]['Conductivity']['Data'], # note, only the first layer of df lowercase... potentially need to make casting recursive in future
                    'inverse temperature' : df["conductivity experiment"]["conductivity data"]['Inverse temperature']['Data'],
                    # Not keeping log conductivity since it gets messed up when taking the mean. Leave it to user to self define
                    #'log conductivity' : df["conductivity experiment"]["conductivity data"]['Log(Conductivity)']['Data'],
                }

                for component in df["electrolyte"]["electrolyte component"]:
                    if component['Acronym'] == None: continue                               # Skip any empty component entries
                    #print(component)
                    # amount in molar units
                    units = component['Substance amount unit']
                    data_dict[component['Acronym'] + '_' + units] = component['Substance amount']
                    # amount by weight
                    units = component['Amount unit']
                    data_dict[component['Acronym'] + '_' + units] = component['Amount']

                ## Non-list entries in data to be repeated in the CSV output (e.g. EC content = 0.2 for conductivity [0.1, 0.2, 0.3])
                for k, v in data_dict.items():
                    if not isinstance(v, list): data_dict[k] = itertools.repeat(v)

                temp_df = pd.DataFrame(zip(*data_dict.values()), columns = list(data_dict.keys()))

                global_df = pd.concat([global_df, temp_df], sort=False)# <-- Adds data_dict entries    

                #Report if file introduced any duplicates
                if duplicate_count != len(global_df.index[global_df.duplicated()]):
                    dif = len(global_df.index[global_df.duplicated()]) - duplicate_count
                    if verbose: print("Warning, importing \'{}{}\' introduced {} duplicate measurement values".format(
                        path_to_json,
                        filename,
                        dif))
                    duplicate_count = len(global_df.index[global_df.duplicated()])

            except KeyError:
                print("Invalid data key for \'{}{}\'\nReturning data for examination".format(
                        path_to_json,
                        filename))
                return df
            except:
                print("Data import failed for \'{}{}\'\nReturning data for examination".format(
                        path_to_json,
                        filename))
                return df


    #Replace NaN values with 0
    global_df.fillna(0., inplace=True)

    #Warn of duplicates and remove
    if duplicate_count > 0 and verbose: print("{} duplicate values detected and removed".format(duplicate_count))
    global_df.drop_duplicates(inplace=True)


    ## Save to csv
    global_df.to_csv(path_to_csv, index=False)
    
    print("Import Summary:\n\tLabels: {}\n\tData points: {}".format(list(global_df.columns), global_df.shape[0]))

    return global_df.reset_index(drop=True)

# simple helper function
def to_list(arg): return arg if isinstance(arg, list) else [arg]

def feature_overview(
        data: pd.DataFrame,
        objective_funcs: Union[str, List[str]],
        features: Union[str, List[str]],
        fig_size: Tuple[int,int] = (6,4),
        save_loc: Union[bool, str] = False) -> None:
    """
    Outputs correlation/covariance plots for features along with feature importance plots for each objective function (uses ``sklearn.ensemble.RandomForestRegressor`` with default hyperparameters)

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements

    objective_funcs : Union[str, List[str]]
        Objective function or list of objective functions

    features : Union[str, List[str]]
        Feature or list of features

    fig_size : Tuple[int,int]
        Size of plots (follows matplotlib convention)

        By default ``(6,4)``.

    save_loc : Union[bool, str]
        Name to save plots (if desired), if ``False`` the plots will only be shown, not saved.

        Saving filename convention is:

        - save_loc + 'corr.pdf'
        - save_loc + 'cov.pdf'
        - save_loc + objective function name + '-feature-importance.pdf'

    Returns
    -------
        ``None``
    """
   
    # Cast to values to list if they're given as a single string
    features = to_list(features)
    objective_funcs = to_list(objective_funcs)

    print('Feature correlation and covariance metrics:')

    ## Generate correlation matrix for imported data
    plt.figure(figsize=fig_size)
    correlation_matrix = data[features].corr()
    sn.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, cmap=mpl.colormaps['coolwarm'], cbar=False, annot_kws={"size": 14})
    plt.title("Correlation", fontsize=20)
    if len(features) > 3:
        plt.xticks([])
        plt.yticks(fontsize=16)
    else:
        plt.yticks([])
        plt.xticks(fontsize=16)
    plt.tight_layout()
    if save_loc: plt.savefig(save_loc + 'corr.pdf', bbox_inches="tight")
    plt.show()

    ## Likewise covariance
    plt.figure(figsize=fig_size)
    covariance_matrix = data[features].cov()
    plt.title("Covariance", fontsize=20)
    max_val = np.max(covariance_matrix.abs().max())
    sn.heatmap(covariance_matrix, annot=True, vmin=-max_val, vmax=max_val, cmap=mpl.colormaps['coolwarm'], cbar=False, annot_kws={"size": 12})
    if len(features) > 3:
        plt.xticks([])
        plt.yticks(fontsize=16)
    else:
        plt.yticks([])
        plt.xticks(fontsize=16)
    plt.tight_layout()
    if save_loc: plt.savefig(save_loc + 'cov.pdf', bbox_inches="tight")
    plt.show()

    # Note: Scaling not necessary for RF
    for obj in to_list(objective_funcs):
        rf = RandomForestRegressor()
        rf.fit(data[features], data[obj])
        feature_importance = rf.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(6, np.max((int(len(features)/3),3))))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(data[features].columns)[sorted_idx], fontsize=16)
        plt.xlim([0,1])
        plt.title('Feature Importance: {}'.format(obj), fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        if save_loc: plt.savefig(save_loc + obj.replace('/','_') + '-feature-importance.pdf', bbox_inches="tight")
        plt.show()

    print("Feature space dimensions: {}, datapoints: {}, objective function(s): {}".format(len(features), data.shape[0], objective_funcs))

def combine_cut(
        data: pd.DataFrame,
        objective_funcs: Union[str, List[str]],
        features: Union[str, List[str]],
        max_samples: int = 5) -> pd.DataFrame:
    """
    Takes the mean value of the given objective functions for measurements where every declared feature is identical and also records the standard deviations. Measurements for identical input features will be combined into a single entry.

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements

    objective_funcs : Union[str, List[str]]
        Objective function or list of objective functions

    features : Union[str, List[str]]
        Feature or list of features

    max_samples : int
        Max number of samples to use to calculate the standard deviation of the mean. We take only the first n values and slice the rest away (i.e. not randomized).

        By default ``5``.

    Returns
    -------
        ``DataFrame``
            DataFrame with the combined mean values for the objective functions, their standard deviations and the declared features. The resulting DataFrame will then only have row entries with unique input features. An additional `count` column is also added to record the number of repeated measurements for that feature set.
    """
   
    # Cast to values to list if they're given as a single string
    features = to_list(features)
    objective_funcs = to_list(objective_funcs)

    # Slice dataframe to only include objective functions and features
    data = pd.concat([data[features], data[objective_funcs]], axis=1)
    
    prior_count = data.shape[0]

    ## Calculate the standard deviation of the mean for each set of measurements for a given composition
    def f(samples):
        if len(samples) > max_samples: samples = samples.iloc[:max_samples] #slice away any extra measurements
        n_samples = len(samples)
        std_mean = samples.std(ddof=1)/np.sqrt(n_samples)
        return std_mean

    # Slices dataframe into groups of rows with all the same features, then calculates the mean value for each objective function and deviations
    # Finally, also counts the # of measurements combined
    data = data.groupby(features).agg(
            **{obj: pd.NamedAgg(column=obj, aggfunc='mean') for obj in objective_funcs},
            **{obj+"_std": pd.NamedAgg(column=obj, aggfunc=lambda x: np.std(x, ddof=1)) for obj in objective_funcs},
            **{obj+"_std_mean": pd.NamedAgg(column=obj, aggfunc=f) for obj in objective_funcs},
            **{'count': pd.NamedAgg(column=features[0], aggfunc='size')}
            ).reset_index()
    
    print("Combined {} datapoints down to {}".format(prior_count, data.shape[0]))

    return data

def interactive_data_visualize(
        df: pd.DataFrame,
        fig: plt.Figure,
        features: List[str],
        show_axes: List[str],
        highlight_data: Optional[pd.DataFrame] = None) -> None:
    """
    Visualize dataset with a 2d or 3d plot. Special formulation to support integration with notebook interactive widgets. Plots the distribution of measurement points within the min-max range of each selected feature.

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements.

    fig : ``plt.Figure``
        Figure object passed to contain plots.

    features : List[str]
        List of all features.

    show_axes : List[str]
        List of features to use as axes for the data distribution plot.

    highlight_data : Optional[``DataFrame``]
        Optional DataFrame of measurements to highlight within the plot (shown as red triangles).

        Default value ``None``.

    Examples
    --------
    .. code-block:: python

        from ipywidgets import interact, widgets
        import matplotlib.pyplot as plt
        from IPython.display import display
        %matplotlib notebook

        # Define which DataFrames to use for included/excluded data
        df = sliced_df
        cut_df = cutoff_df

        # Initialize plt.Figure object
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot()

        # Define and show interactive widgets
        X_slider=widgets.Dropdown(options=features, description='X')
        Y_slider=widgets.Dropdown(options=features, description='Y')
        Z_slider=widgets.Dropdown(options=[None]+features, description='Z (optional)')
        display(X_slider)    
        display(Y_slider)    
        display(Z_slider)

        # Function call to update plot
        def data_vis(change=None):
            ax.clear()
            X_feature = X_slider.value
            Y_feature = Y_slider.value
            Z_feature = Z_slider.value
            if Z_feature:
                prep.interactive_data_visualize(df, fig, features, show_axes=[X_feature, Y_feature, Z_feature], highlight_data=cut_df)
            else:
                prep.interactive_data_visualize(df, fig, features, show_axes=[X_feature, Y_feature], highlight_data=cut_df)


        # Make button to plot data
        btn=widgets.Button(description="Run")
        display(btn)
        btn.on_click(data_vis)

    Returns
    -------
        ``None``
    """
    fig.clf()
    colorscheme = mpl.colormaps['winter']
    show_axes = list(dict.fromkeys(show_axes)) # only support unique axes, but maintain order
    if len(show_axes) > 3 or len(show_axes) < 2: raise Exception("Supported dimensions: 3, 2")

    data = df.groupby(show_axes).agg(**{'count': pd.NamedAgg(column=show_axes[0], aggfunc='size')}).reset_index()

    if len(show_axes) == 3:
        ax = fig.add_subplot(111, projection='3d')
        x = data[show_axes[0]]
        y = data[show_axes[1]]
        z = data[show_axes[2]]
        c = data['count']


        other_dims = [dim for dim in features if dim not in show_axes]
        if len(other_dims) > 0:
            ax.set_title("Dataset summary", fontsize=18)
            print("Data in dimensions {} colored by number of values.\nTotal datapoints: {}".format(other_dims, df.shape[0]))
            if c.max() - c.min() != 0:
                img = ax.scatter(x, y, z, c=c, cmap=colorscheme, alpha=0.5)
                fig.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.04)
            else:
                img = ax.scatter(x, y, z, c='black', alpha=0.5)
        else:
            ax.set_title("Dataset summary", fontsize=18)
            print("Total datapoints: {}".format(df.shape[0]))
            img = ax.scatter(x, y, z, c="black", alpha=0.5)
        ax.set_xlabel(x.name, fontsize=12)
        ax.set_ylabel(y.name, fontsize=12)
        ax.set_zlabel(z.name, fontsize=12)
        if isinstance(highlight_data, pd.DataFrame):
            data = highlight_data.groupby(show_axes).agg(**{'count': pd.NamedAgg(column=show_axes[0], aggfunc='size')}).reset_index()
            x = data.iloc[:,0]
            y = data.iloc[:,1]
            z = data.iloc[:,2]
            img2 = ax.scatter(x, y, z, c='red', marker='^', alpha=1)

    elif len(show_axes) == 2:
        ax = fig.add_subplot(111)
        x = data[show_axes[0]]
        y = data[show_axes[1]]
        c = data['count']
        other_dims = [dim for dim in features if dim not in show_axes]
        if len(other_dims) > 0:
            ax.set_title("Dataset summary", fontsize=18)
            print("Data in dimensions {} colored by number of values.\nTotal datapoints: {}".format(other_dims, df.shape[0]))
            if c.max() - c.min() != 0:
                img = ax.scatter(x, y, c=c, cmap=colorscheme, alpha=0.5)
                fig.colorbar(img, orientation='vertical', fraction=0.046, pad=0.04)
            else:
                img = ax.scatter(x, y, c='black', alpha=0.5)
        else:
            ax.set_title("Dataset summary", fontsize=18)
            print("Total datapoints: {}".format(df.shape[0]))
            img = ax.scatter(x, y, c="black")

        ax.set_xlabel(x.name, fontsize=16)
        ax.set_ylabel(y.name, fontsize=16)
        if isinstance(highlight_data, pd.DataFrame):
            data = highlight_data.groupby(show_axes).agg(**{'count': pd.NamedAgg(column=show_axes[0], aggfunc='size')}).reset_index()
            x = data.iloc[:,0]
            y = data.iloc[:,1]
            img2 = ax.scatter(x, y, c='red', marker='^', alpha=1)


 


    ax.tick_params(labelsize=16)
    plt.show()

def data_visualize(
        df: pd.DataFrame,
        features: List[str],
        show_axes: List[str]) -> None:
    """
    Visualize dataset with a 2d or 3d plot. Plots the distribution of measurement points within the min-max range of each selected feature.

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements.

    features : List[str]
        List of all features.

    show_axes : List[str]
        List of features to use as axes for the data distribution plot.

    Returns
    -------
        ``None``
    """
    fig = plt.figure()
    colorscheme = mpl.colormaps['winter']
    if len(show_axes) > 3 or len(show_axes) < 2: raise Exception("Supported dimensions: 3, 2")

    data = df.groupby(show_axes).agg(**{'count': pd.NamedAgg(column=show_axes[0], aggfunc='size')}).reset_index()
    if len(show_axes) == 3:
        ax = fig.add_subplot(111, projection='3d')
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        z = data.iloc[:,2]
        c = data.iloc[:,3]
        other_dims = [dim for dim in features if dim not in show_axes]
        if len(other_dims) > 0:
            ax.set_title("Dataset summary")
            print("Data in dimensions {} colored by number of values.\nTotal datapoints: {}".format(other_dims, df.shape[0]))
            img = ax.scatter(x, y, z, c=c, cmap=colorscheme, alpha=0.5)
            if c.max() - c.min() != 0: fig.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.04)
        else:
            ax.set_title("Dataset summary")
            print("Total datapoints: {}".format(df.shape[0]))
            img = ax.scatter(x, y, z, c="black", alpha=0.5)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_zlabel(z.name)
    elif len(show_axes) == 2:
        ax = fig.add_subplot(111)
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        c = data.iloc[:,2]
        other_dims = [dim for dim in features if dim not in show_axes]
        if len(other_dims) > 0:
            ax.set_title("Dataset summary")
            print("Data in dimensions {} colored by number of values.\nTotal datapoints: {}".format(other_dims, df.shape[0]))
            img = ax.scatter(x, y, c=c, cmap=colorscheme, alpha=0.5)
            if c.max() - c.min() != 0: fig.colorbar(img, orientation='vertical', fraction=0.046, pad=0.04)
        else:
            ax.set_title("Dataset summary")
            print("Total datapoints: {}".format(df.shape[0]))
            img = ax.scatter(x, y, c="black")

        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
 


    plt.show()

def outlier_filter(
        data: pd.DataFrame,
        objective_funcs: Union[str, List[str]],
        cluster_dimensions: Union[str, List[str]],
        filter_threshhold: float = 0.98,
        quantile_filter: bool = False,
        min_cluster_size: int = 5,
        show_plots: bool = True) -> pd.DataFrame:
    """
    Data outlier detection using `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/index.html>`_  clustering algorithm. HDBSCAN requires only one parameter (``min_cluster_size``) and we use the resulting outlier_scores for the datapoints we fit it on. Most important parameter: ``cluster_dimensions`` defines the data the clusterer actually considers, i.e. [objective function] if you just want to filter outlier measured results, or [objective function] + feature_list if you want to cluster in all dimensions. The function will also plot kept vs filtered data points for each feature to help with tuning parameters.

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements.

    objective_funcs : Union[str, List[str]]
        List of the objective functions - used for plotting each of the objective as a function of the features considered for clustering

    cluster_dimensions : Union[str, List[str]]
        Defines which dimensions will be considered by the HDBSCAN clustering algorithm. Takes label names from `data` columns.
    
    filter_threshhold : float (range [0-1])
        Outlier score threshold value. Datapoints with an outlier score higher than this value will be filtered.

        Default value ``0.98``.

    quantile_filter : bool
        - quantile_filter = ``True``: keep data under filter_threshhold*100 percentile outlier score (from total dataset)
        - quantile_filter = ``False``: keep data under filter_threshhold outlier score

        Default value ``False``

    min_cluster_size : int
        HDBSCAN parameter for choosing ``min_cluster_size``, see `HDBSCAN documentation <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-min-cluster-size>`_.

        Default value ``5``.

    show_plots: bool
        Whether to show plots of objective_functions : cluster_dimensions

        Default value ``True``

    Returns
    -------
        ``DataFrame``
            DataFrame with outlier values removed
    """

    # Cast to values to list if they're given as a single string
    cluster_dimensions = to_list(cluster_dimensions)
    objective_funcs = to_list(objective_funcs)

    # Initialize hdbscan clustering obj and run
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(data[cluster_dimensions])

    # Extract outlier scores from hdbscan clusterer
    outlier_scores = pd.Series(clusterer.outlier_scores_, name='outlier score')
    data = pd.concat([data, outlier_scores], axis=1)

    if quantile_filter:
        outlier_filter = np.where(clusterer.outlier_scores_ > outlier_scores.quantile(filter_threshhold))[0]
    else: outlier_filter = np.where((clusterer.outlier_scores_ > filter_threshhold))[0]

    #Plots
    if show_plots:
        for obj in objective_funcs:
            for feature in data[cluster_dimensions].columns:
                plt.scatter(data[feature], data[obj], linewidth=0, c='gray', alpha=0.25)
                plt.scatter(data[feature][outlier_filter], data[obj][outlier_filter], linewidth=0, c='red', alpha=0.5)
                plt.xlabel(feature)
                plt.ylabel(obj)
                plt.show()

    print("Number of datapoints filtered out: %d, (%d%% of dataset)" % (
        len(outlier_filter),
        len(outlier_filter)/data.shape[0]*100)
    )
    display(data.loc[outlier_filter]) # display for pretty notebook printing
    return data.loc[~data.index.isin(outlier_filter)].reset_index(drop=True)


def filter_fn(
        data: pd.DataFrame,
        cols: Union[str, List[str]],
        func: Callable[..., bool]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the provided function to each value in the given column(s). If the function returns true, that measurement is removed from the filtered DataFrame (and added to the `caught` DataFrame).

    Parameters
    ----------
    data : ``DataFrame``
        DataFrame of experimental measurements.

    cols: Union[str, List[str]]
        List of columns in DataFrame to apply filter function to.

    func: Callable[..., bool]
        Function taking one argument (each value in the selected columns) and returning boolean (``True`` indicating to remove value).

    Returns
    -------
        Tuple[``DataFrame``, ``DataFrame``]
            The first ``DataFrame`` is the `caught` values which are removed from the second ``DataFrame``, which is the filtered (clean) DataFrame.
    """

    cols = to_list(cols)

    removed_values = pd.DataFrame()

    for col in cols:
        removed_values = pd.concat([removed_values, data.loc[func(data[col])]], axis=0)
        data = data.loc[~func(data[col])]

    print("{} measurements removed".format(removed_values.shape[0]))
    display(removed_values)
    return removed_values, data.reset_index(drop=True)

def arrhenius(
        data: pd.DataFrame, feature_list: Union[str, List[str]], objective: str = 'conductivity',
        inverse_temp: str = 'inverse temperature', min_samples: int = 5, beta_0: Optional[float] = None,
        n_fits: int = 50, random_state: Optional[int] = None,
        save_loc: Union[str, bool] = False
    ) -> Tuple[float, List[str], List[str], pd.DataFrame]:
    """
    Transform objective function into Arrhenius fitted surrogate model (:math:`log(\\sigma) \\rightarrow S_0, S_1, S_2`). This function expects the objective function along with standard deviations in the form
    of a DataFrame with columns: e.g. `"inverse temperature", "X", "conductivity", "conductivity_std"`, where `"X"` can be some arbitrary feature set.
    This transformation groups all data with identical `"X"` features into a single datapoint with the objective functions :math:`S_0, S_1, S_2`,
    and makes the inverse temperature implicit.

    For each unique composition (read: identical `"X"` values):
        - the data should contain a single row with the average value and deviations of repeated measurements of the objective function at each inverse temperature.
        - This function randomly perturbs the average measured value for that composition **for each inverse temperature** in the form :math:`log(perturbed) = log(objective) \\cdot (1 + random\_normal(std))` and applies the Arrhenius fit.
        - This is repeated for `n_fits` (by default 50 times). The resulting average coefficient values, their standard deviations and metrics for the quality of the fit on all measurements with this composition are returned.
    
    The Arrhenius surrogate model has the form:

    .. math:: log_{10}(objective) = S_0 - S_1 (\\beta - \\beta_0) - S_2 (\\beta - \\beta_0)^2

    :math:`\\beta` values should have 1000/T\[K\] scale, and :math:`\\beta_0` can be freely chosen (if not user defined, the function searches for
    the :math:`\\beta_0` value between 1000/(273.15-50) and 1000/(273.15+100) which results in minimal correlation between the :math:`S_0` and :math:`S_1` 
    coefficients. The returned DataFrame includes the surrogate :math:`S_{0,1,2}` objective functions as well as their deviations (`S0_std`, etc.). 
    The results DataFrame also includes metrics for the quality of the Arrhenius fits:
    `'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)'`

    If no :math:`\\beta_0` value is provided, by default a plot of the :math:`S_0 : S_1` correlation from -50 - 100C is generated.

    In addition, an overview of the Arrhenius fit quality is provided by displaying the compositions with the top 10% log(MAE) as well as a histogram
    overview of the log(MAE) for the whole dataset.

    **Disambiguation: This function should be used to process a DataFrame with combined rows (-> mean value, deviations) for repeated measurement**
    (i.e. 1 row for 5 measurements of the conductivity of an electrolyte with composition `"X"` and inverse temperature 
    :math:`\\beta` -> `"conductivity", "conductivity_std"`)

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements.

    feature_list : Untion[str, List[str]]
        Feature or list of features

    objective : str
        `data` column label of objective function to use for the Arrhenius fits (only supports single objective function).

        Default value ``conductivity``.

    inverse_temp: str
        `data` column label of the inverse temperature (in 1000/T\[K\] scale) for the measured values.
        The `data` DataFrame should be properly prepared to have this information before calling this function.

        Default value ``inverse temperature``.

    min_samples: int
        Minimum number of measurements for the feature set excluding inverse temperature (i.e. how many different temperatures were measured for a given formulation). Compositions below this threshold are discarded.

        Default value ``5``.

    beta_0: Optional[float]
        This value corresponds to the temperature where `S_0` and `S_1` become uncorrelated. If no value is provided, the function will automatically search for the lowest correlated temperature from -50C to 100C (in intervals of 5C).

        Default value ``None``.

    n_fits : int
        Number of times to vary the objective function (+- random normal perturbation based on the standard deviations of the measurement) and refit to estimate coefficient uncertainty.

        Default value ``50``

    random_state : Optional[int]
        Sets a numpy random seed for reproducibility.

        Default value ``None``.

    save_loc : Union[bool, str]
        Name to save plot (if desired), if ``False`` the plot will only be shown, not saved.

        Saving filename convention is:
        save_loc + 'onset_temp_plot.pdf'

    Returns
    -------
        Tuple[float, List[str], List[str], pd.DataFrame]
            4 tuple of:

            - float (beta_0),
            - List[str] (feature list),
            - List[str] (S0 S1 S2 objective functions list),
            - DataFrame with added S0, S0_std, S1, S1_std, S2, S2_std and 'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)' columns
    """
        ## Create groups for Arrhenius fit by grouping by features - temp
    rng = np.random.default_rng(seed=random_state)

    min_T = 1000/data[inverse_temp].max()-273.15
    max_T = 1000/data[inverse_temp].min()-273.15
    inv_temps = 1000/(np.arange(-50, 100+1, 5)+273.15)
    default_beta_0 = 1000/273.15
    correlation = dict(beta_0=[],corr=[])
    average_std = data[objective + "_std"].dropna().mean()

    def coef_shift(group_df, beta_0, beta_n): # Shift S0, S1 coefficients from beta_0 to beta_n
        group_df = group_df.copy()
        S0, S1, S2 = group_df['S0'], group_df['S1'], group_df['S2']
        if beta_0 == beta_n:
            return group_df
        else:
            S0_n = S0 - S1*(beta_n - beta_0) - S2*(beta_n - beta_0)**2
            S1_n = (S0_n - S0) / (beta_0 - beta_n) - S2*(beta_0 - beta_n)
            group_df['S0'] = S0_n
            group_df['S1'] = S1_n
            return group_df

    def perturbed_fit(group_df): # Full fit with perturbation to estimate S_0,1,2 deviation
        if group_df.shape[0] < min_samples: return None # only take the data with n=min_samples or more measured points
        ## Create linear regression
        LR = LinearRegression()
        X = pd.DataFrame({
            'x' : group_df[inverse_temp] - default_beta_0,
            'x**2' : (group_df[inverse_temp] - default_beta_0)**2.
            })
        y = group_df[objective]
        y_std = group_df[objective + "_std"].fillna(average_std) # assume average variation of std of mean for values with only 1 sample
        LR.fit(X, np.log10(y)) # Arrhenius -> log of objective

        # Generate n_fits perturbed datasets to estimate S_x uncertainties
        perturbed_coeffs = {'S0':[],'S1':[],'S2':[],'MAE':[]}
        for _ in range(n_fits):
            perturb_LR = LinearRegression()
            #log10_perturb_y = np.nan_to_num(np.log10(y + np.random.normal(np.zeros(len(y)),y_std_mean)), nan=-1.79769313e+308) ## catch case where perturbation leads to a negative conductivity (log10 = largest negative supported float)
            log10_perturb_y = np.nan_to_num(np.log10(y)*(1+rng.normal(np.zeros(len(y)),y_std)), nan=-1.79769313e+308) ## Newer version according to Heuer recommendation log(pert) = log(sigma)*(1+noise)
            perturb_LR.fit(X, log10_perturb_y) # Arrhenius -> log of objective
            perturbed_coeffs['S0'].append(perturb_LR.intercept_)
            perturbed_coeffs['S1'].append(-1*perturb_LR.coef_[0])
            perturbed_coeffs['S2'].append(-1*perturb_LR.coef_[1])
            perturbed_coeffs['MAE'].append(mean_absolute_error(log10_perturb_y, perturb_LR.predict(X)))
        ## Add values under the respective coefficient names
        # Need to collapse the groups now to only return composition : S0 / S1 / S2 
        MAE = np.mean(perturbed_coeffs['MAE'])
        MAE = -float('inf') if MAE == 0 else MAE
        result = pd.Series({
                'S0': np.mean(perturbed_coeffs['S0']),
                'S0_std': np.std(perturbed_coeffs['S0'], ddof=1),
                'S1': np.mean(perturbed_coeffs['S1']),
                'S1_std': np.std(perturbed_coeffs['S1'], ddof=1),
                'S2': np.mean(perturbed_coeffs['S2']),
                'S2_std': np.std(perturbed_coeffs['S2'], ddof=1),
                'log(MAE arrh fit)': np.log(MAE)
                })

        # Calculate MAE for Arrhenius fit prediction of input samples
        def arrh_back(beta):
            return np.power(10, result['S0'] - result['S1']*(beta - default_beta_0) - result['S2']*(beta - default_beta_0)**2)
        MARE = mean_absolute_error(group_df[objective]/arrh_back(group_df[inverse_temp]), np.ones(group_df.shape[0]))
        MSRE = mean_squared_error(group_df[objective]/arrh_back(group_df[inverse_temp]), np.ones(group_df.shape[0]))
        #MAE = np.log(mean_absolute_error(group_df[objective],np.log10(arrh_back(group_df[inverse_temp]))))
        MARE = -float('inf') if MARE == 0 else MARE
        MSRE = -float('inf') if MSRE == 0 else MSRE
        result['Mean Absolute (Relative) Error'] = MARE
        result['Mean Squared (Relative) Error'] = MSRE

        return pd.concat([group_df[arrhenius_group].iloc[0], result])

    arrhenius_group = feature_list.copy()
    arrhenius_group.remove(inverse_temp)
    ## Redefine objective function to multidimensional:
    objective_list = ['S0', 'S1', 'S2']
    further_info = ['S0_std', 'S1_std', 'S2_std', 'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)']

    ## passes dataframe of group matching the groupby criteria (all vals are equal in these columns) to function arrhenius_fit
    data = data.groupby(arrhenius_group).apply(perturbed_fit).dropna().reset_index(drop=True)
    data = data[arrhenius_group + objective_list + further_info]
    
    # Show S0 / S1 correlation as a function of selected beta_0
    if beta_0 == None:
        #calculate beta_0 with minimal S0:S1 correlation
        ## passes dataframe of group matching the groupby criteria (all vals are equal in these columns) to function arrhenius_fit
        for beta_n in inv_temps:
            temp_data = coef_shift(data, default_beta_0, beta_n)
            if temp_data.empty: continue
            correlation['beta_0'].append(beta_n)
            correlation['corr'].append(temp_data.loc[:,['S0','S1']].corr().iloc[0,1])


        beta_0 = correlation['beta_0'][np.argmin(np.abs(correlation['corr']))] # select beta_0 with minimal S0 / S1 coef correlation
        print("beta_0 temperature: {}".format(1000/beta_0 - 273.15))
        plt.plot(figsize=(8,4))
        plt.plot((1000/np.array(correlation['beta_0'])-273.15), correlation['corr'])
        plt.xlabel(r"T $\degree$C", fontsize=18)
        plt.ylabel("Correlation", fontsize=18)
        plt.title("S0, S1 correlation", fontsize=20)
        plt.axvline(1000/beta_0-273.15, c='red', linestyle='--')
        plt.axhline(0, c='black', linestyle='--')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        if save_loc: plt.savefig(save_loc + 'onset_temp_plot.pdf')
        plt.show()


    data = coef_shift(data, default_beta_0, beta_0) # finalize S0 / S1 / S2 coeffs to either manually selected or min-correlated value

    print("Upper quantile log Arrhenius error:")
    display(data.loc[data['log(MAE arrh fit)'] > data['log(MAE arrh fit)'].quantile(0.9)])

    #Display histogram of fit error
    plt.plot(figsize=(8,4))
    plt.hist(data['log(MAE arrh fit)'])
    plt.ylabel('N(Compositions)')
    plt.xlabel('log(MAE arrh fit)')
    if save_loc: plt.savefig(save_loc + 'arrhenius_MAE_hist.pdf')
    plt.show()

    return beta_0, arrhenius_group, objective_list, data

def direct_sample_arrhenius(
        data: pd.DataFrame, feature_list: Union[str, List[str]], objective: str = 'conductivity',
        max_error: Optional[float] = None,
        inverse_temp: str = 'inverse temperature', min_samples: int = 5, beta_0: Optional[float] = None,
        n_fits: int = 50, random_state: Optional[int] = None,
        save_loc: Union[str, bool] = False
    ) -> Tuple[float, List[str], List[str], pd.DataFrame]:
    """
    Transform objective function into Arrhenius fitted surrogate model (:math:`log(\\sigma) \\rightarrow S_0, S_1, S_2`). This function expects repeated measurements of the objective function in the form
    of a DataFrame with columns: e.g. `"inverse temperature", "X", "conductivity", where `"X"` can be some arbitrary feature set.

    For each unique composition (read: identical `"X"` values):
        - the data should contain several repeated measurements of the objective function at varying inverse temperatures.
        - This function randomly selects a single measurement for that composition **for each inverse temperature** and applies the Arrhenius fit.
        - This is repeated for `n_fits` (by default 50 times). The resulting average coefficient values, their standard deviations and metrics for the quality of the fit on all measurements with this composition are returned.
    
    The Arrhenius surrogate model has the form:

    .. math:: log_{10}(objective) = S_0 - S_1 (\\beta - \\beta_0) - S_2 (\\beta - \\beta_0)^2

    :math:`\\beta` values should have 1000/T\[K\] scale, and :math:`\\beta_0` can be freely chosen (if not user defined, the function searches for
    the :math:`\\beta_0` value between 1000/(273.15-50) and 1000/(273.15+100) which results in minimal correlation between the :math:`S_0` and :math:`S_1` 
    coefficients. The returned DataFrame includes the surrogate :math:`S_{0,1,2}` objective functions as well as their deviations (`S0_std`, etc.). 
    The results DataFrame also includes metrics for the quality of the Arrhenius fits:
    `'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)'`

    If no :math:`\\beta_0` value is provided, by default a plot of the :math:`S_0 : S_1` correlation from -50 - 100C is generated.

    In addition, an overview of the Arrhenius fit quality is provided by displaying the compositions with the top 10% log(MAE) as well as a histogram
    overview of the log(MAE) for the whole dataset.

    **Disambiguation: This function should be used to process a DataFrame with individual rows for each repeated measurement**
    (i.e. 5 rows for 5 measurements of the conductivity of an electrolyte with composition `"X"` and inverse temperature :math:`\\beta`)

    Parameters
    ----------
    data : ``DataFrame``
        Dataframe of experimental measurements.

    feature_list : Untion[str, List[str]]
        Feature or list of features

    objective : str
        `data` column label of objective function to use for the Arrhenius fits (only supports single objective function).

        Default value ``conductivity``.

    inverse_temp: str
        `data` column label of the inverse temperature (in 1000/T\[K\] scale) for the measured values.
        The `data` DataFrame should be properly prepared to have this information before calling this function.

        Default value ``inverse temperature``.

    min_samples: int
        Minimum number of measurements for the feature set excluding inverse temperature (i.e. how many different temperatures were measured for a given formulation). Compositions below this threshold are discarded.

        Default value ``5``.

    beta_0: Optional[float]
        This value corresponds to the temperature where `S_0` and `S_1` become uncorrelated. If no value is provided, the function will automatically search for the lowest correlated temperature from -50C to 100C (in intervals of 5C).

        Default value ``None``.

    n_fits : int
        Number of times to vary the objective function (+- random normal perturbation based on the standard deviations of the measurement) and refit to estimate coefficient uncertainty.

        Default value ``50``

    random_state : Optional[int]
        Sets a numpy random seed for reproducibility.

        Default value ``None``.

    save_loc : Union[bool, str]
        Name to save plot (if desired), if ``False`` the plot will only be shown, not saved.

        Saving filename convention is:
        save_loc + 'onset_temp_plot.pdf'

    Returns
    -------
        Tuple[float, List[str], List[str], pd.DataFrame]
            4 tuple of:

            - float (beta_0),
            - List[str] (feature list),
            - List[str] (S0 S1 S2 objective functions list),
            - DataFrame with added S0, S0_std, S1, S1_std, S2, S2_std and 'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)' columns
    """
    rng = np.random.default_rng(seed=random_state)

    min_T = 1000/data[inverse_temp].max()-273.15
    max_T = 1000/data[inverse_temp].min()-273.15
    inv_temps = 1000/(np.arange(-50, 100+1, 5)+273.15)
    default_beta_0 = 1000/273.15
    correlation = dict(beta_0=[],corr=[])
    arrhenius_group = feature_list.copy()
    arrhenius_group.remove(inverse_temp)

    def coef_shift(group_df, beta_0, beta_n): # Shift S0, S1 coefficients from beta_0 to beta_n
        group_df = group_df.copy()
        S0, S1, S2 = group_df['S0'], group_df['S1'], group_df['S2']
        if beta_0 == beta_n:
            return group_df
        else:
            S0_n = S0 - S1*(beta_n - beta_0) - S2*(beta_n - beta_0)**2
            S1_n = (S0_n - S0) / (beta_0 - beta_n) - S2*(beta_0 - beta_n)
            group_df['S0'] = S0_n
            group_df['S1'] = S1_n
            return group_df

    def perturbed_fit(group_df): # Full fit with perturbation to estimate S_0,1,2 deviation
        inv_temps = group_df[inverse_temp].unique()
        if len(inv_temps) < min_samples: return None # only take the data with n=min_samples or more measured points
        # We create a list of sub-dataframes with each df having matching inverse temperatures
        grouped_inv_temps = [group_df.loc[group_df['inverse temperature'] == inv_temp] for inv_temp in inv_temps]

        # Generate n_fits randomly sampled datasets to estimate S_x and uncertainties
        perturbed_coeffs = {'S0':[],'S1':[],'S2':[], 'MAE':[]}
        for _ in range(n_fits):
            perturb_LR = LinearRegression()

            # Generate our randomly selected set of measurements (1 per inv_temp)
            sampled_df = pd.DataFrame()
            for df in grouped_inv_temps:
                sampled_df = pd.concat([sampled_df, df.sample(random_state=rng)]) # appends a randomly sampled measurement for each inv_temp

            y = sampled_df[objective]
            X = pd.DataFrame({
                'x' : sampled_df[inverse_temp] - default_beta_0,
                'x**2' : (sampled_df[inverse_temp] - default_beta_0)**2.
                })

            log10_perturb_y = np.log10(y)
            perturb_LR.fit(X, log10_perturb_y) # Arrhenius -> log of objective
            perturbed_coeffs['S0'].append(perturb_LR.intercept_)
            perturbed_coeffs['S1'].append(-1*perturb_LR.coef_[0])
            perturbed_coeffs['S2'].append(-1*perturb_LR.coef_[1])
            perturbed_coeffs['MAE'].append(mean_absolute_error(log10_perturb_y, perturb_LR.predict(X)))
        
        # Need to collapse the groups now to only return composition : S0 / S1 / S2 
        MAE = np.mean(perturbed_coeffs['MAE'])
        MAE = -float('inf') if MAE == 0 else MAE
        result = pd.Series({
                'S0': np.mean(perturbed_coeffs['S0']),
                'S0_std': np.std(perturbed_coeffs['S0'], ddof=1),
                'S1': np.mean(perturbed_coeffs['S1']),
                'S1_std': np.std(perturbed_coeffs['S1'], ddof=1),
                'S2': np.mean(perturbed_coeffs['S2']),
                'S2_std': np.std(perturbed_coeffs['S2'], ddof=1),
                'log(MAE arrh fit)': np.log(MAE)
                })

        # Calculate MAE for Arrhenius fit prediction of input samples
        def arrh_back(beta):
            return np.power(10, result['S0'] - result['S1']*(beta - default_beta_0) - result['S2']*(beta - default_beta_0)**2)
        MARE = mean_absolute_error(group_df[objective]/arrh_back(group_df[inverse_temp]), np.ones(group_df.shape[0]))
        MSRE = mean_squared_error(group_df[objective]/arrh_back(group_df[inverse_temp]), np.ones(group_df.shape[0]))
        #MAE = np.log(mean_absolute_error(group_df[objective],np.log10(arrh_back(group_df[inverse_temp]))))
        MARE = -float('inf') if MARE == 0 else MARE
        MSRE = -float('inf') if MSRE == 0 else MSRE
        result['Mean Absolute (Relative) Error'] = MARE
        result['Mean Squared (Relative) Error'] = MSRE

        return pd.concat([group_df[arrhenius_group].iloc[0], result])

    ## Redefine objective function to multidimensional:
    objective_list = ['S0', 'S1', 'S2']
    further_info = ['S0_std', 'S1_std', 'S2_std', 'Mean Absolute (Relative) Error', 'Mean Squared (Relative) Error', 'log(MAE arrh fit)']

    ## passes dataframe of group matching the groupby criteria (all vals are equal in these columns) to function arrhenius_fit
    data = data.groupby(arrhenius_group).apply(perturbed_fit).dropna().reset_index(drop=True)
    data = data[arrhenius_group + objective_list + further_info]
    
    # Show S0 / S1 correlation as a function of selected beta_0
    if beta_0 == None:
        #calculate beta_0 with minimal S0:S1 correlation
        ## passes dataframe of group matching the groupby criteria (all vals are equal in these columns) to function arrhenius_fit
        for beta_n in inv_temps:
            temp_data = coef_shift(data, default_beta_0, beta_n)
            if temp_data.empty: continue
            correlation['beta_0'].append(beta_n)
            correlation['corr'].append(temp_data.loc[:,['S0','S1']].corr().iloc[0,1])


        beta_0 = correlation['beta_0'][np.argmin(np.abs(correlation['corr']))] # select beta_0 with minimal S0 / S1 coef correlation
        print("beta_0 temperature: {}".format(1000/beta_0 - 273.15))
        plt.plot(figsize=(8,4))
        plt.plot((1000/np.array(correlation['beta_0'])-273.15), correlation['corr'])
        plt.xlabel(r"T $\degree$C", fontsize=18)
        plt.ylabel("Correlation", fontsize=18)
        plt.title("S0, S1 correlation", fontsize=20)
        plt.axvline(1000/beta_0-273.15, c='red', linestyle='--')
        plt.axhline(0, c='black', linestyle='--')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        if save_loc: plt.savefig(save_loc + 'direct_onset_temp_plot.pdf')
        plt.show()

    data = coef_shift(data, default_beta_0, beta_0) # finalize S0 / S1 / S2 coeffs to either manually selected or min-correlated value

    print("Upper quantile Arrhenius fit error:")
    display(data.loc[data['Mean Absolute (Relative) Error'] > data['Mean Absolute (Relative) Error'].quantile(0.9)])

    #Display histogram of fit error
    plt.plot(figsize=(8,4))
    plt.hist(data['log(MAE arrh fit)'])
    plt.ylabel('N(Compositions)')
    plt.xlabel('log(MAE arrh fit)')
    plt.tight_layout()
    if save_loc: plt.savefig(save_loc + 'direct_arrhenius_MAE_hist.pdf')
    plt.show()

    return beta_0, arrhenius_group, objective_list, data
