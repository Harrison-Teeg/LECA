# Author: Mirko Fischer
# Date: 12.08.2024
# Version: 0.0.1
# License: MIT license

import pickle
from LECA import fit
from LECA.prep import to_list
from typing import List, Tuple, Union, Optional, Callable, Dict
from PyALAF.multi_optimize_pool import run_batch_learning_multi
from PyALAF.multi_optimize import run_continuous_batch_learning_multi
from PyALAF.optimize import run_batch_learning, run_continuous_batch_learning
from PyALAF.aggregation_fn import identity_aggregation_fn
from PyALAF.models import PoolModel

import numpy as np
import pandas as pd
import copy
from copy import deepcopy
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def initialize_dataset(pool=None):
    pass

def features_to_composition():
    pass

class ActiveLearner:
    """
    Interface object between LECAs WorkFlow object and the PyAL library. 

    Parameters
    ----------
    wf : LECA WorkFlow object
        An initially trained LECA workflow, providing data and model information.
    data_pool: Optional[Union[pd.DataFrame, np.ndarray]]
        A pool of data that may be considered for Active Learning. If ``None`` is provided, population-based active
        learning will be performed instead of pool-based active learning.

        Default value ``None``.

    Attributes
    ----------
    wf: LECA WorkFlow object
        An initially trained LECA workflow, providing data and model information.

    active_set_X: pd.DataFrame
        Features of the data set used for active learning. The models are trained based on these data and further data is selected.

    active_set_y: pd.DataFrame
        Objectives for the active learning set.

    test_set_X: pd.DataFrame
        Features for a test set, similar to the WorkFlows validation holdout set.

    test_set_y: pd.DataFrame
        Objectives for a test set, similar to the WorkFlows validation holdout set.

    X: pd.DataFrame
        Features for the active set and test set together.
    
    y: pd.DataFrame
        Objectives for the active set and test set together.

    data_pool: Union[pd.DataFrame, np.ndarray]
        A pool of data that may be considered for Active Learning. If a pool is provided, data from this pool
        might be added to the active set. If ``None`` is provided, population-based active
        learning will be performed instead of pool-based active learning.

    objective_funcs: List[str]
        List of all objective functions.

    estimator_names: List[str]
        Name of all estimators used in the LECA workflow.
    """
    def __init__(self, wf: fit.WorkFlow, data_pool: Optional[Union[pd.DataFrame, np.ndarray]]=None) -> None:
        self.wf = wf

        self.scaler = wf.scaler

        #Copy data because we might modify it and do not want to modify the workflow directly
        self.active_set_X = copy.deepcopy(self.wf.X_unscaled)
        self.active_set_y = copy.deepcopy(self.wf.y)

        try:
            self.test_set_X = copy.deepcopy(self.wf.X_validate_unscaled)
            self.test_set_y = copy.deepcopy(self.wf.y_validate)
        except:
            self.test_set_X = pd.DataFrame()
            self.test_set_y = pd.DataFrame()

        self.X = pd.concat([self.active_set_X, self.test_set_X])
        self.y = pd.concat([self.active_set_y, self.test_set_y])

        self.data_pool = data_pool

        self.objective_funcs = list(self.wf.y.columns)
        self.estimator_names = {}
        for obj in self.objective_funcs:
            self.estimator_names[obj] = list(self.wf.results[obj].keys())

        if hasattr(self.wf, "polynomial_degree"):
            self.polynomial_degree = self.wf.polynomial_degree
        else:
            self.polynomial_degree = 3

        self.suggested_data = None


    def data_importance(self, 
                        estimators: Optional[Union[str, List[str]]] = None,
                        objective_funcs: Optional[Union[str, List[str]]] = None, 
                        acquisition_function: Optional[Union[str, List[str]]]='ideal',
                        aggregation_function: Optional[callable]=None, 
                        lim_features: Optional[List[float]]=[-1,1],
                        repeat: int=1, initial_samples: int=10,  alpha: Union[float, List[float]]=10.0,
                        shuffle_sets=False, random_state: Optional[int] = None,
                        initialization: Optional[str]='random', training_subset=None, select_test_from_sets_equally=True, cv=None, 
                        sets_for_selection=None,
                        **kwargs):
        
        """
        Perform an analysis of how much data of all collected data is really needed. An initial model is trained 
        and afterwards sequentially data points are added to the pool of training data. Metrics are evaluated for 
        each step.

        Parameters
        ----------
        estimators: Optional[Union[str, List[str]]]
            String or list with model name(s) to perform datasize performance analysis. ``None`` will use all workflow models.

            Default value ``None``.

        objective_funcs: Optional[Union[str, List[str]]]
            String or list of strings declaring which objective functions on which to perform datasize performance analysis.
            When ``None`` is passed, defaults to all objective functions.
            If a list of objective functions are passed, the function returns a list of objects.

            Default value ``None``.

        acquisition_function: Optional[Union[str, List[str]]]
            String or list of strings declaring which acquisition functions should be used. The active learning is performed
            for each of the listed acquisition functions. 
            
            Valid acquisition functions for GPR models are: ucb, poi, ei, GSx, GSy, iGS, ideal, qbc, std

            Valid acquisition functions for non-GPR models are: GSx, GSy, iGS, ideal, qbc

            Default value ``ideal``

        aggregation_function: Optional[Callable]
            If several objectives are given data may be selected based on the improvement for all objectives instead of
            only one objective. This function defines how different objectives are combined to a single objective.
            It must have ``uncert`` as a boolean parameter, which defines how to handle the calculation of the uncertainty 
            for the combined objective, which might be different to the calculation of the combined objective itself.
            If ``None`` is provided, active learning will be performed for each objective seperately.

            Default value ``None``

        repeat: Optional[int]
            Number of times to repeat the active learning.

            Default value ``1``.

        initial_samples: Optional[int]
            Number of data points used to fit an initial model.

            Default value ``10``.

        alpha: Optional[Union[float, List[float]]]
            Hyperparameter for the active learning algorithm. 
            For following acquisition functions a hyperparameter is used:
            ei: Weighing of exploration vs. exploitation
            ideal: IDEAL hyperparameter
            qbc: Number of models in the ensemble

            Although other models ignore this parameter a value must be given.

            Default value ``10``.

        suffle_sets : Optional[bool]
            When True the trainining and validation holdout set from the workflow are mixed. 
            Use this, when the result depends extremely on the chosen validation holdout set.
            
            Default value ``False``.

        training_subset : Optional[pd.DataFrame]
            You should always use a workflow containing all data available. To use only a subset of data for training
            the models for datasize performans, you can specify training_subset.
            This way, the test set will always be selected from all available data, even if the models are only
            trained on part of the data. This ensures consistency in test errors. 
            
            Example:
            We assume data was collected during 3 Active Learning iterations. Data named Gen1 corresponds to the first, 
            Gen2 to the second and Gen3 to the third iteration. The initial data is named Gen0. 
            We will always take data from Gen0 to Gen3 for testing. 
            Let's say we want to evaluate the improvement of the models when adding Gen2 data. 
            This means, we use Gen0 and Gen1 as initial data (Initial samples needs to be specified, 
            together with initialization='data'). Then the datasize_performance is only run for Gen2.

            If you want to perform a similar analysis for Gen3, the test errors will be consistent.

            Default value ``None``.

        random_state : Optional[int]
            Sets a numpy random seed for reproducibility.

            Default value ``None``.


        Returns
        -------
        score_dict: Dict
            Dictionary with active learning results
        """

        def find_matching_indices(df, column_names, reference_values):
                                # Check if inputs have correct length
                                if len(column_names) != len(reference_values):
                                    raise ValueError("Number of column names must match number of reference values")
                                
                                # Create boolean mask for matching indices
                                mask = np.ones(len(df), dtype=bool)
                                for idx, ref_val in zip(column_names, reference_values):
                                    mask &= (np.isclose(df[:, idx], ref_val, atol=1e-10))
                                
                                # Return indices where all conditions are met
                                return np.where(mask)[0].tolist()

        if objective_funcs == None: objective_funcs = self.objective_funcs
        rng = np.random.default_rng(seed=random_state)

        score_dict={}

        if select_test_from_sets_equally:
            if isinstance(sets_for_selection, list):
                print('Selecting data from provided extra sets equally')
            else:
                if initialization == 'data' and isinstance(training_subset, pd.DataFrame):
                    print('Selecting data for test set from initial set, training subset and the rest equally.')
                    rest_idx = []
                    for sample in np.array(training_subset):
                        l = find_matching_indices(np.array(self.X), np.arange(0, training_subset.shape[1]), sample)
                        if len(l) > 0:
                            rest_idx += l
                    rest_idx = [l for l in range(len(self.X)) if l not in rest_idx]
                    X_rest = self.X.iloc[np.array(rest_idx)]

                    rest_idx = []
                    for sample in np.array(initial_samples):
                        l = find_matching_indices(np.array(X_rest), np.arange(0, initial_samples.shape[1]), sample)
                        if len(l) > 0:
                            rest_idx += l
                    rest_idx = [l for l in range(len(X_rest)) if l not in rest_idx]
                    X_rest = X_rest.iloc[np.array(rest_idx)]

                    rest_idx = []
                    for sample in np.array(initial_samples):
                        l = find_matching_indices(np.array(training_subset), np.arange(0, initial_samples.shape[1]), sample)
                        if len(l) > 0:
                            rest_idx += l
                    rest_idx = [l for l in range(len(training_subset)) if l not in rest_idx]
                    training_subset = training_subset.iloc[np.array(rest_idx)]

                elif initialization != 'data' and isinstance(training_subset, pd.DataFrame):
                    print('Selecting data for test set from training subset and the rest equally.')
                    rest_idx = []
                    for sample in np.array(training_subset):
                        l = find_matching_indices(np.array(self.X), np.arange(0, training_subset.shape[1]), sample)
                        if len(l) > 0:
                            rest_idx += l
                    rest_idx = [l for l in range(len(self.X)) if l not in rest_idx]
                    X_rest = self.X.iloc[np.array(rest_idx)]

                elif initialization == 'data' and not isinstance(training_subset, pd.DataFrame):
                    print('Selecting data for test set from initial set and the rest equally.')
                    rest_idx = []
                    for sample in np.array(initial_samples):
                        l = find_matching_indices(np.array(self.X), np.arange(0, initial_samples.shape[1]), sample)
                        if len(l) > 0:
                            rest_idx += l
                    rest_idx = [l for l in range(len(self.X)) if l not in rest_idx]
                    X_rest = self.X.iloc[np.array(rest_idx)]

                else:
                    print('Selecting data for test set from full data set randomly.')
                    X_rest = self.X
            

        #evaluate each objective separately
        if aggregation_function == None:
            #Simply use identity aggregation function if no aggregation function is provided
            #This is slower but this way also full functionality is provided
            aggregation_function == identity_aggregation_fn

            #for obj in to_list(objective_funcs):
            #    print('Active Learning for objective: {}'.format(obj))
            #    if estimators == None: estimators = self.estimator_names[obj]
            #    score_dict[obj] = {}
            #
            #    for model in to_list(estimators):
            #        print('Model: {}'.format(model))
            #        score_dict[obj][model] = {}
            #        for acf, alpha_a in zip(to_list(acquisition_function), to_list(alpha)):
            #            print('Acquisition function: {}'.format(acf))
            #            score_dict[obj][model][acf] = {}
            #            for i in range(repeat):
            #                print('Iteration {}/{}'.format(i+1, repeat))
            #                random_state_act = random_state+i if random_state != None else None

            #                estimator = Pipeline([('scaler', clone(self.scaler)),
            #                                     ('model', clone(self.wf.get_estimator(model, obj)))]) 
            #                evaluation_model = PoolModel(features=self.X, objective=self.y[obj])
                            
            #                if shuffle_sets:
            #                    active_set, test_set = train_test_split(np.array(self.X),
            #                                                            test_size = len(self.test_set_X),
            #                                                            random_state = random_state_act,
            #                                                            shuffle=True)
            #                else:
            #                    active_set =np.array(self.active_set_X)
            #                    test_set = np.array(self.test_set_X)
            #                print(lim_features)
            #                samples, observation_y, result = run_batch_learning(evaluation_model, 
            #                regression_model=estimator,
            #                acquisition_function = acf,
            #                pool = active_set, 
            #                batch_size = 1,
            #                noise=0.0,
            #                initial_samples=initial_samples, 
            #                active_learning_steps=len(active_set)-initial_samples,
            #                lim=None,
            #                alpha=alpha_a,
            #                random_state=random_state_act,
            #                return_samples=False,
            #                initialization=initialization,
            #                test_set = test_set,
            #                poly_degree = self.polynomial_degree,
            #                fictive_noise_level = 0,
            #                calculate_test_metrics = True
            #                )

            #                score_dict[obj][model][acf]['iteration {}'.format(i)] = {}
            #                score_dict[obj][model][acf]['iteration {}'.format(i)]['samples'] = samples
            #                score_dict[obj][model][acf]['iteration {}'.format(i)]['result'] = result

            #return score_dict

        #Evaluate all objectives together
        #else:

        score_dict = {}
        initial_samples_original = deepcopy(initial_samples)

        if estimators == None: estimators = self.estimator_names[obj[0]]
        for model in to_list(estimators):
            print('Model: {}'.format(model))
            score_dict[model] = {}

            current_estimators = []
            evaluation_models = []
            for obj in to_list(objective_funcs):
                wf_estimator = clone(self.wf.get_estimator(model, obj))
                print('Old random state:', wf_estimator.random_state)
                wf_estimator.random_state = random_state
                wf_estimator.solver = 'lbfgs'
                print('New random state:',wf_estimator.random_state)

                #We clone the estimator. This should be a pure estimator model without application of any scaler
                #The scaler must be defined separately
                #All data is scaled by default using the MinMaxScaler, which is fitted to the limits
                estimator = clone(wf_estimator) #Pipeline([('scaler', clone(self.scaler)),
                                        #('model', wf_estimator)]) 
                print(estimator)
                current_estimators.append(estimator)
                evaluation_model = PoolModel(features=self.X, objective=self.y[obj])
                evaluation_models.append(evaluation_model)
            
            for acf, alpha_a in zip(to_list(acquisition_function), to_list(alpha)):
                print('Acquisition function: {}'.format(acf))
                score_dict[model][acf] = {}


                if cv == None:
                    print('Using randomized test sets (if no test set is given)')
                    #Repetitions start here
                    ######################################################################################################
                    for i in range(repeat):
                        print('Iteration {}/{}'.format(i+1, repeat))
                        random_state_act = random_state+i if random_state != None else None
                        print('Random State: {}'.format(random_state))

                        #First, select test set from all data
                        #This way, we make sure that the test data is taken from all available data,
                        #even if we fit models only using data up to a certain generation of AL
                        #e.g. use data from Gen0 + Gen1 for training and data from Gen0 + Gen1 + Gen2 for testing
                        #Furthermore, we may later also select Gen0 as initial data (see below)
                        if select_test_from_sets_equally:
                            if isinstance(sets_for_selection, list):
                                n_subset = len(self.test_set_X)/len(self.X)
                                a_list = []
                                t_list = []
                                for s, set in enumerate(sets_for_selection):
                                    a, t = train_test_split(np.array(set),
                                                                test_size = n_subset,
                                                                random_state = random_state_act+s,
                                                                shuffle=True)
                                    a_list.append(a)
                                    t_list.append(t)
                                active_set = np.vstack(a_list)
                                test_set = np.vstack(t_list)
                                
                                #Second, select a training subset from the active set
                                if isinstance(training_subset, pd.DataFrame):
                                    active_set_indices = []
                                    for sample in np.array(training_subset):
                                        l = find_matching_indices(active_set, np.arange(0, training_subset.shape[1]), sample)
                                        if len(l) > 0:
                                            active_set_indices += l
                                    active_set = active_set[np.array(active_set_indices)]

                                #Third, select the initial data points
                                #The initial data points must be in the training subset
                                if initialization == 'data':
                                    initial_sample_indices = []
                                    for sample in np.array(initial_samples_original):
                                        l = find_matching_indices(active_set, np.arange(0, initial_samples_original.shape[1]), sample)
                                        if len(l) > 0:
                                            initial_sample_indices += l

                                    initial_samples = np.array(initial_sample_indices)
                                    #print(initial_samples)
                                    active_learning_steps = len(active_set)-len(initial_samples)
                                    #print(active_learning_steps)
                                else:
                                    active_learning_steps = len(active_set)-initial_samples

                            else:
                                if not shuffle_sets:
                                    raise Exception('Shuffle_sets must be `True` to use this option')
                                elif initialization == 'data' and isinstance(training_subset, pd.DataFrame):
                                    n_subset = len(self.test_set_X)/len(self.X)
                                    a1, t1 = train_test_split(np.array(training_subset),
                                                                test_size = n_subset,
                                                                random_state = random_state_act,
                                                                shuffle=True)
                                    a2, t2 = train_test_split(np.array(initial_samples_original),
                                                                test_size = n_subset,
                                                                random_state = random_state_act,
                                                                shuffle=True)
                                    if len(X_rest) > 0:
                                        a3, t3 = train_test_split(np.array(X_rest),
                                                                    test_size = n_subset,
                                                                    random_state = random_state_act,
                                                                    shuffle=True)
                                        active_set = np.vstack([a2, a1])
                                        test_set = np.vstack([t1, t2, t3])
                                    else:
                                        active_set = np.vstack([a2, a1])
                                        test_set = np.vstack([t1, t2])

                                    initial_samples = np.arange(len(a2))
                                    #print(initial_samples)
                                    active_learning_steps = len(active_set)-len(initial_samples)
                                    
                                    
                                elif initialization != 'data' and isinstance(training_subset, pd.DataFrame):
                                    n_subset = len(self.test_set_X)/len(self.X)
                                    a1, t1 = train_test_split(np.array(training_subset),
                                                                test_size = n_subset,
                                                                random_state = random_state_act,
                                                                shuffle=True)
                                    
                                    if len(X_rest) > 0:
                                        a2, t2 = train_test_split(np.array(X_rest),
                                                                    test_size = n_subset,
                                                                    random_state = random_state_act,
                                                                    shuffle=True)
                                        
                                        active_set = a1
                                        test_set = np.vstack([t1, t2])
                                    else:
                                        active_set = a1
                                        test_set = t1

                                    active_learning_steps = len(active_set)-initial_samples
                                

                                elif initialization == 'data' and not isinstance(training_subset, pd.DataFrame):
                                    n_subset = len(self.test_set_X)/len(self.X)
                                    a1, t1 = train_test_split(np.array(initial_samples_original),
                                                                test_size = n_subset,
                                                                random_state = random_state_act,
                                                                shuffle=True)
                                    a2, t2 = train_test_split(np.array(X_rest),
                                                                test_size = n_subset,
                                                                random_state = random_state_act,
                                                                shuffle=True)
                                    
                                    active_set = np.vstack([a1, a2])
                                    test_set = np.vstack([t1, t2])

                                    initial_samples = np.arange(len(a1))
                                    active_learning_steps = len(active_set)-len(initial_samples)
                                    

                                else:
                                    active_set, test_set = train_test_split(np.array(self.X),
                                                                            test_size = len(self.test_set_X),
                                                                            random_state = random_state_act,
                                                                            shuffle=True)
                                    
                                    active_learning_steps = len(active_set)-initial_samples


                        else:
                            if shuffle_sets:
                                active_set, test_set = train_test_split(np.array(self.X),
                                                                        test_size = len(self.test_set_X),
                                                                        random_state = random_state_act,
                                                                        shuffle=True)
                    
                            else:
                                active_set =np.array(self.active_set_X)
                                test_set = np.array(self.test_set_X)

                            #Second, select a training subset from the active set
                            if isinstance(training_subset, pd.DataFrame):
                                active_set_indices = []
                                for sample in np.array(training_subset):
                                    l = find_matching_indices(active_set, np.arange(0, training_subset.shape[1]), sample)
                                    if len(l) > 0:
                                        active_set_indices += l
                                active_set = active_set[np.array(active_set_indices)]

                            #Third, select the initial data points
                            #The initial data points must be in the training subset
                            if initialization == 'data':
                                initial_sample_indices = []
                                for sample in np.array(initial_samples_original):
                                    l = find_matching_indices(active_set, np.arange(0, initial_samples_original.shape[1]), sample)
                                    if len(l) > 0:
                                        initial_sample_indices += l

                                initial_samples = np.array(initial_sample_indices)
                                #print(initial_samples)
                                active_learning_steps = len(active_set)-len(initial_samples)
                                #print(active_learning_steps)
                            else:
                                active_learning_steps = len(active_set)-initial_samples

                        print('Active Set: {}'.format(len(active_set)))
                        print('Test Set: {}'.format(len(test_set)))
                        print('Initial samples: {}'.format(initial_samples))
                        print('Active Learning Steps: {}'.format(active_learning_steps))

                        samples, observation_y, result = run_batch_learning_multi(evaluation_models,
                        aggregation_function=aggregation_function,
                        regression_models=current_estimators,
                        acquisition_function = acf,
                        pool = active_set,
                        batch_size=1,
                        noise=0,
                        lim_features=lim_features,
                        feature_scaler='min_max',
                        initial_samples=initial_samples,
                        active_learning_steps=active_learning_steps,
                        alpha=alpha_a,
                        initialization=initialization,
                        test_set=test_set,
                        random_state=random_state_act,
                        calculate_test_metrics=True,
                        **kwargs)

                        score_dict[model][acf]['iteration {}'.format(i)] = {}
                        score_dict[model][acf]['iteration {}'.format(i)]['samples'] = samples
                        score_dict[model][acf]['iteration {}'.format(i)]['observation'] = observation_y
                        score_dict[model][acf]['iteration {}'.format(i)]['active_set'] = active_set
                        score_dict[model][acf]['iteration {}'.format(i)]['test_set'] = test_set
                        score_dict[model][acf]['iteration {}'.format(i)]['random_state'] = random_state_act
                        score_dict[model][acf]['iteration {}'.format(i)]['result agg'] = result['aggregated']
                        for j, obj in enumerate(to_list(objective_funcs)):
                            score_dict[model][acf]['iteration {}'.format(i)]['result {}'.format(obj)] = result['model_{}'.format(j)]

                    ######################################################################################################

                else:
                    from sklearn.model_selection import KFold
                    print('Using cross-validation based sets (if no test set is given)')
                    print('Initial data points are chosen randomly from the training set')
                    print('For each of the {} training sets {} different random initializations are chosen'.format(cv, repeat))
                    i=0
                    
                    #Repetitions start here
                    ######################################################################################################
            

                    #First, select test set from all data
                    #This way, we make sure that the test data is taken from all available data,
                    #even if we fit models only using data up to a certain generation of AL
                    #e.g. use data from Gen0 + Gen1 for training and data from Gen0 + Gen1 + Gen2 for testing
                    #Furthermore, we may later also select Gen0 as initial data (see below)
                    if select_test_from_sets_equally:
                        
                        if isinstance(sets_for_selection, list):
                            n_subset = len(self.test_set_X)/len(self.X)
                            fold = KFold(n_splits=cv, shuffle=True, random_state = random_state)
                            active_folds = []
                            test_folds = []
                            a_list = []
                            t_list = []
                            for s, set in enumerate(sets_for_selection):
                                a1_list = []
                                t1_list = []
                                for _, (train_index, test_index) in enumerate(fold.split(set)):
                                    a1 = set[train_index]
                                    t1 = set[test_index]
                                    a1_list.append(a1)
                                    t1_list.append(t1)

                                a_list.append(a1_list)
                                t_list.append(t1_list)                            

                            for c in range(cv):
                                a_rel_list = []
                                t_rel_list = []
                                for s, set in enumerate(sets_for_selection):
                                    a_rel_list.append(a_list[s][c])
                                    t_rel_list.append(t_list[s][c])

                                active = np.vstack(a_rel_list)
                                test = np.vstack(t_rel_list)
                                print('Size of Sets in fold {}'.format(c))
                                print(len(active))
                                print(len(test))

                                active_folds.append(active)
                                test_folds.append(test)    

                        else:
                            if not shuffle_sets:
                                raise Exception('Shuffle_sets must be `True` to use this option')
                            elif initialization == 'data' and isinstance(training_subset, pd.DataFrame):
                                fold = KFold(n_splits=cv, shuffle=True, random_state = random_state)
                                active_folds = []
                                test_folds = []
                                training_subset_arr = np.array(training_subset)
                                init_arr = np.array(initial_samples_original)
                                rest_arr = np.array(X_rest)
                                a1_list = []
                                a2_list = []
                                a3_list = []
                                t1_list = []
                                t2_list = []
                                t3_list = []
                                for _, (train_index, test_index) in enumerate(fold.split(training_subset_arr)):
                                    a1 = training_subset_arr[train_index]
                                    t1 = training_subset_arr[test_index]
                                    a1_list.append(a1)
                                    t1_list.append(t1)

                                for _, (train_index, test_index) in enumerate(fold.split(training_subset_arr)):
                                    a2 = training_subset_arr[train_index]
                                    t2 = training_subset_arr[test_index]
                                    a2_list.append(a2)
                                    t2_list.append(t2)

                                if len(X_rest) > 0:
                                    for _, (train_index, test_index) in enumerate(fold.split(rest_arr)):
                                        a3 = init_arr[train_index]
                                        t3 = init_arr[test_index]
                                        a3_list.append(a3)
                                        t3_list.append(t3)
                                
                                for c in range(cv):
                                    a1 = a1_list[c]
                                    t1 = t1_list[c]
                                    a2 = a2_list[c]
                                    t2 = t2_list[c]

                                    if len(X_rest) > 0:
                                        a3 = a3_list[c]
                                        t3 = t3_list[c]
                                        active = np.vstack([a2, a1])
                                        test = np.vstack([t1, t2, t3])
                                        
                                    else:
                                        active = np.vstack([a2, a1])
                                        test = np.vstack([t1, t2])

                                    active_folds.append(active)
                                    test_folds.append(test)    
                            
                            
                            elif initialization != 'data' and isinstance(training_subset, pd.DataFrame):
                                fold = KFold(n_splits=cv, shuffle=True, random_state = random_state)
                                active_folds = []
                                test_folds = []
                                training_subset_arr = np.array(training_subset)
                                rest_arr = np.array(X_rest)
                                a1_list = []
                                a2_list = []
                                t1_list = []
                                t2_list = []
                                for _, (train_index, test_index) in enumerate(fold.split(training_subset_arr)):
                                    a1 = training_subset_arr[train_index]
                                    t1 = training_subset_arr[test_index]
                                    a1_list.append(a1)
                                    t1_list.append(t1)

                                if len(X_rest) > 0:
                                    for _, (train_index, test_index) in enumerate(fold.split(rest_arr)):
                                        a2 = init_arr[train_index]
                                        t2 = init_arr[test_index]
                                        a2_list.append(a2)
                                        t2_list.append(t2)
                                
                                for c in range(cv):
                                    a1 = a1_list[c]
                                    t1 = t1_list[c]


                                    if len(X_rest) > 0:
                                        a2 = a2_list[c]
                                        t2 = t2_list[c]
                                        active = a1
                                        test = np.vstack([t1, t2])
                                        
                                    else:
                                        active = a1
                                        test = t1
                                    active_folds.append(active)
                                    test_folds.append(test)                        

                            elif initialization == 'data' and not isinstance(training_subset, pd.DataFrame):
                                fold = KFold(n_splits=cv, shuffle=True, random_state = random_state)
                                active_folds = []
                                test_folds = []
                                init_arr = np.array(initial_samples_original)
                                rest_arr = np.array(X_rest)
                                a1_list = []
                                a2_list = []
                                t1_list = []
                                t2_list = []
                                for _, (train_index, test_index) in enumerate(fold.split(init_arr)):
                                    a1 = init_arr[train_index]
                                    t1 = init_arr[test_index]
                                    a1_list.append(a1)
                                    t1_list.append(t1)

                                for _, (train_index, test_index) in enumerate(fold.split(rest_arr)):
                                    a2 = init_arr[train_index]
                                    t2 = init_arr[test_index]
                                    a2_list.append(a1)
                                    t2_list.append(t1)
                                
                                for c in range(cv):
                                    a1 = a1_list[c]
                                    a2 = a2_list[c]
                                    t1 = t1_list[c]
                                    t2 = t2_list[c]

                                    active = np.vstack([a1, a2])
                                    test = np.vstack([t1, t2])
                                    active_folds.append(active)
                                    test_folds.append(test)                            

                            else:
                                fold = KFold(n_splits=cv, shuffle=False)
                                active_folds = []
                                test_folds = []
                                X_arr = np.array(self.X)
                                for _, (train_index, test_index) in enumerate(fold.split(X_arr)):
                                    active = X_arr[train_index]
                                    test = X_arr[test_index]
                                    active_folds.append(active)
                                    test_folds.append(test)


                    else:
                        if shuffle_sets:
                            fold = KFold(n_splits=cv, shuffle=shuffle_sets, random_state = random_state)
                            active_folds = []
                            test_folds = []
                            X_arr = np.array(self.X)
                            for _, (train_index, test_index) in enumerate(fold.split(X_arr)):
                                active = X_arr[train_index]
                                test = X_arr[test_index]
                                active_folds.append(active)
                                test_folds.append(test)
                
                        else:
                            raise Exception('CV can not be used when using fixed active set')
                            active_set =np.array(self.active_set_X)
                            test_set = np.array(self.test_set_X)

                    
                    for c in range(cv):
                        active_set = active_folds[c]
                        test_set = test_folds[c]


                        for kk in range(repeat):
                            print('Iteration {}/{}'.format(i+1, int(cv*repeat)))
                            random_state_act = random_state+i if random_state != None else None
                            print('Random State: {}'.format(random_state))

                            if select_test_from_sets_equally:
                                if isinstance(sets_for_selection, list):
                                    #Second, select a training subset from the active set
                                    if isinstance(training_subset, pd.DataFrame):
                                        active_set_indices = []
                                        for sample in np.array(training_subset):
                                            l = find_matching_indices(active_set, np.arange(0, training_subset.shape[1]), sample)
                                            if len(l) > 0:
                                                active_set_indices += l
                                        active_set = active_set[np.array(active_set_indices)]

                                    #Third, select the initial data points
                                    #The initial data points must be in the training subset
                                    if initialization == 'data':
                                        initial_sample_indices = []
                                        for sample in np.array(initial_samples_original):
                                            l = find_matching_indices(active_set, np.arange(0, initial_samples_original.shape[1]), sample)
                                            if len(l) > 0:
                                                initial_sample_indices += l

                                        initial_samples = np.array(initial_sample_indices)
                                        #print(initial_samples)
                                        active_learning_steps = len(active_set)-len(initial_samples)
                                        #print(active_learning_steps)
                                    else:
                                        active_learning_steps = len(active_set)-initial_samples

                                else:

                                    if initialization == 'data' and isinstance(training_subset, pd.DataFrame):
                                        initial_samples = np.arange(len(a2))
                                        #print(initial_samples)
                                        active_learning_steps = len(active_set)-len(initial_samples)

                                    elif initialization != 'data' and isinstance(training_subset, pd.DataFrame):
                                        active_learning_steps = len(active_set)-initial_samples

                                    elif initialization == 'data' and not isinstance(training_subset, pd.DataFrame):
                                        initial_samples = np.arange(len(a1_list[c]))
                                        active_learning_steps = len(active_set)-len(initial_samples)

                                    else:
                                        active_learning_steps = len(active_set)-initial_samples
                            
                            else:
                                #Second, select a training subset from the active set
                                if isinstance(training_subset, pd.DataFrame):
                                    active_set_indices = []
                                    for sample in np.array(training_subset):
                                        l = find_matching_indices(active_set, np.arange(0, training_subset.shape[1]), sample)
                                        if len(l) > 0:
                                            active_set_indices += l
                                    active_set = active_set[np.array(active_set_indices)]

                                #Third, select the initial data points
                                #The initial data points must be in the training subset
                                if initialization == 'data':
                                    initial_sample_indices = []
                                    for sample in np.array(initial_samples_original):
                                        l = find_matching_indices(active_set, np.arange(0, initial_samples_original.shape[1]), sample)
                                        if len(l) > 0:
                                            initial_sample_indices += l

                                    initial_samples = np.array(initial_sample_indices)
                                    #print(initial_samples)
                                    active_learning_steps = len(active_set)-len(initial_samples)
                                    #print(active_learning_steps)
                                else:
                                    active_learning_steps = len(active_set)-initial_samples

                            print('Active Set: {}'.format(len(active_set)))
                            print('Test Set: {}'.format(len(test_set)))
                            print('Initial samples: {}'.format(initial_samples))
                            print('Active Learning Steps: {}'.format(active_learning_steps))

                            samples, observation_y, result = run_batch_learning_multi(evaluation_models,
                            aggregation_function=aggregation_function,
                            regression_models=current_estimators,
                            acquisition_function = acf,
                            pool = active_set,
                            batch_size=1,
                            noise=0,
                            lim_features=lim_features,
                            feature_scaler='min_max',
                            initial_samples=initial_samples,
                            active_learning_steps=active_learning_steps,
                            alpha=alpha_a,
                            initialization=initialization,
                            test_set=test_set,
                            random_state=random_state_act,
                            calculate_test_metrics=True,
                            **kwargs)

                            score_dict[model][acf]['iteration {}'.format(i)] = {}
                            score_dict[model][acf]['iteration {}'.format(i)]['samples'] = samples
                            score_dict[model][acf]['iteration {}'.format(i)]['observation'] = observation_y
                            score_dict[model][acf]['iteration {}'.format(i)]['active_set'] = active_set
                            score_dict[model][acf]['iteration {}'.format(i)]['test_set'] = test_set
                            score_dict[model][acf]['iteration {}'.format(i)]['random_state'] = random_state_act
                            score_dict[model][acf]['iteration {}'.format(i)]['result agg'] = result['aggregated']
                            for j, obj in enumerate(to_list(objective_funcs)):
                                score_dict[model][acf]['iteration {}'.format(i)]['result {}'.format(obj)] = result['model_{}'.format(j)]
                            i+=1
                ######################################################################################################

        return score_dict
        
    
    def single_step_al(self, 
                        estimator: Optional[str] = None,
                        objective_funcs: Optional[Union[str, List[str]]] = None, 
                        acquisition_function: Optional[Union[str, List[str]]]='ideal',
                        aggregation_function: Optional[callable]=None, 
                        alpha: float=10.0, lim:Optional[np.ndarray]=None, batch_size:int=10,
                        random_state: Optional[int] = None, opt_method: Optional[str] = 'PSO',
                        **kwargs):
        
        if objective_funcs == None: objective_funcs = self.objective_funcs
        rng = np.random.default_rng(seed=random_state)

        if aggregation_function == None:
            print('An aggregation function is NOT used.')
            #Evaluate for each objective function
            for obj in to_list(objective_funcs):
                print('Active Learning for objective: {}'.format(obj))
                if estimator == None: estimator = self.wf.best_model[obj]

                #Get a cloned model
                model = clone(self.wf.get_estimator(estimator, obj))
                
                #Interactively enter new data
                evaluation_model = InteractiveModel(features=self.X, objective=self.y[obj])

                #Pool based learning
                if isinstance(self.data_pool, np.ndarray):

                    #Not supported right now
                    #samples, result = run_batch_learning(evaluation_model, 
                    #    regression_model=model,
                    #    acquisition_function = acquisition_function,
                    #    pool = self.data_pool, 
                    #    batch_size = batch_size,
                    #    noise=0.0,
                    #    initial_samples=np.concatenate([self.active_set_X, self.test_set_X]),
                    #    active_learning_steps=1,
                    #    lim=None,
                    #    alpha=alpha,
                    #    random_state=random_state,
                    #    return_samples=True,
                    #    initialization='data',
                    #    test_set = None,
                    #    poly_degree = self.polynomial_degree,
                    #    fictive_noise_level = 0,
                    #    calculate_test_metrics = False)
                    pass
                #Population based learning
                else:
                    #Not supported right now
                    #samples, result = run_continuous_batch_learning(evaluation_model, 
                    #    regression_model=model,
                    #    acquisition_function = acquisition_function,
                    #    batch_size = batch_size,
                    #    noise=0.0,
                    #    initial_samples=np.array(self.X),
                    #    active_learning_steps=1,
                    #    lim=lim,
                    #    alpha=alpha,
                    #    random_state=random_state,
                    #    return_samples=True,
                    #    initialization='data',
                    #    poly_degree = self.polynomial_degree,
                    #    fictive_noise_level = 0,
                    #    calculate_test_metrics = False)
                    pass
        else:
            print('Using an aggregation function.')
            current_estimators = []
            evaluation_models = []
            for obj in to_list(objective_funcs):
                model = clone(self.wf.get_estimator(estimator, obj))
                current_estimators.append(model)
                evaluation_model = PoolModel(features=self.X, objective=self.y[obj])
                evaluation_models.append(evaluation_model)

            if isinstance(self.data_pool, np.ndarray):
                #Not supported right now
                print('Assuming pool-based learning.')
                pass

            else:
                print('Assuming population-based learning.')
                sample_x, obs_dict, result = run_continuous_batch_learning_multi(evaluation_models, 
                aggregation_function, 
                current_estimators,
                acquisition_function = acquisition_function,
                opt_method = opt_method,
                batch_size = batch_size,
                noise=0,
                #Consider whole data set here, because otherwise AL might suggest to evaluate already evaluated data next
                initial_samples=np.array(self.X), 
                active_learning_steps=1,
                lim_features=lim,
                alpha=alpha,
                random_state=random_state,
                initialization='data',
                poly_degree = self.polynomial_degree,
                calculate_test_metrics=False,
                verbose=False,
                single_update=True,
                **kwargs
                )
                self.suggested_data = sample_x[len(self.X):]
                return sample_x, obs_dict 
            
    
    def get_suggested_data(self):
        return self.suggested_data
    
    def add_suggested_data(self, y):
        if not isinstance(self.suggested_data, np.ndarray):
            raise Exception("There is no suggested data for which objective values can be added")

        if len(self.suggested_data) != len(y):
            raise Exception("The provided data needs to be of the length of the suggested data")
        
        x = pd.DataFrame(self.suggested_data, columns=self.X.columns)
        y = pd.DataFrame(y, columns=self.y.columns)

        self.active_set_X = pd.concat([self.active_set_X, x]).reset_index(drop=True)
        self.active_set_y = pd.concat([self.active_set_y, y]).reset_index(drop=True)

        self.X = pd.concat([self.X, x]).reset_index(drop=True)
        self.y = pd.concat([self.y, y]).reset_index(drop=True)

        self.suggested_data = None

    def update_wf():
        '''
        Extend the training set of the LECA Workflow
        '''
        raise Exception('This method is not implemented yet')

    def automatic_al():
        '''
        Perform automatic active learning and labeling of data
        '''
        raise Exception('This method is not implemented yet')


class InteractiveModel():
    '''
    Gets an initial database. When evaluation data is queryed that is not in the database, the model asks the user 
    to provide that data.
    '''
    def __init__(self, features, objective):
        self.n_features=np.array(features).shape[1]
        self.features=np.array(features)
        self.objective = np.array(objective).flatten()

    def evaluate(self, grid, **kwargs):
        grid = np.array(grid)
        if len(grid.shape)==1:
            grid = grid.reshape(1,-1)
        idx = []
        for i in range(len(grid)):
            index = np.where(np.sum(self.features, axis=1)==np.sum(grid[i]))
            if len(index[0]) != 0:
                index = index[0][0]
                idx.append(index)
            else: 
                user_input = input("Provide Input for {}".format(grid[i]))
                print(self.features)
                print(grid[i])
                self.features = np.vstack([self.features, grid[i]])
                self.objective = np.hstack([self.objective, np.array(user_input)])
                index = np.where(np.sum(self.features, axis=1)==np.sum(grid[i]))[0][0]
                idx.append(index)

        idx = np.array(idx)
        return self.objective[idx]
    
        


