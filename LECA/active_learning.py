
import pickle
from LECA import fit
from LECA.prep import to_list
from typing import List, Tuple, Union, Optional, Callable, Dict
from PyAL.multi_optimize_pool import run_batch_learning_multi
from PyAL.optimize import run_batch_learning
from PyAL.models import PoolModel

import numpy as np
import pandas as pd
import copy
from sklearn.base import clone

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

        #Copy data because we might modify it and do not want to modify the workflow directly
        self.active_set_X = copy.deepcopy(self.wf.X)
        self.active_set_y = copy.deepcopy(self.wf.y)

        self.test_set_X = copy.deepcopy(self.wf.X_validate)
        self.test_set_y = copy.deepcopy(self.wf.y_validate)

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


    def data_importance(self, 
                        estimators: Optional[Union[str, List[str]]] = None,
                        objective_funcs: Optional[Union[str, List[str]]] = None, 
                        acquisition_function: Optional[Union[str, List[str]]]='ideal',
                        aggregation_function: Optional[callable]=None, 
                        repeat: int=1, initial_samples: int=10,  alpha: Union[float, List[float]]=10.0, 
                        random_state: Optional[int] = None,
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

        random_state : Optional[int]
            Sets a numpy random seed for reproducibility.

            Default value ``None``.


        Returns
        -------
        score_dict: Dict
            Dictionary with active learning results
        """

        if objective_funcs == None: objective_funcs = self.objective_funcs
        rng = np.random.default_rng(seed=random_state)

        score_dict={}

        #evaluate each objective separately
        if aggregation_function == None:

            for obj in to_list(objective_funcs):
                print('Active Learning for objective: {}'.format(obj))
                if estimators == None: estimators = self.estimator_names[obj]
                score_dict[obj] = {}

                for model in to_list(estimators):
                    print('Model: {}'.format(model))
                    score_dict[obj][model] = {}
                    for acf, alpha_a in zip(to_list(acquisition_function), to_list(alpha)):
                        print('Acquisition function: {}'.format(acf))
                        score_dict[obj][model][acf] = {}
                        for i in range(repeat):
                            random_state_act = random_state+i if random_state != None else None
                            
                            estimator = clone(self.wf.get_estimator(model, obj))
                            evaluation_model = PoolModel(features=self.X, objective=self.y[obj])

                            samples, result = run_batch_learning(evaluation_model, 
                            regression_model=estimator,
                            acquisition_function = 'ideal',
                            pool = np.array(self.active_set_X), 
                            batch_size = 1,
                            noise=0.0,
                            initial_samples=initial_samples, 
                            active_learning_steps=len(self.active_set_X)-initial_samples,
                            lim=None,
                            alpha=alpha_a,
                            random_state=random_state_act,
                            return_samples=False,
                            initialization='random',
                            test_set = np.array(self.test_set_X),
                            poly_degree = self.polynomial_degree,
                            fictive_noise_level = 0,
                            calculate_test_metrics = True
                            )
                            score_dict[obj][model][acf]['iteration {}'.format(i)] = {}
                            score_dict[obj][model][acf]['iteration {}'.format(i)]['samples'] = samples
                            score_dict[obj][model][acf]['iteration {}'.format(i)]['result'] = result

            return score_dict

        #Evaluate all objectives together
        else:

            score_dict = {}

            if estimators == None: estimators = self.estimator_names[obj[0]]
            for model in to_list(estimators):
                print('Model: {}'.format(model))
                score_dict[model] = {}

                current_estimators = []
                evaluation_models = []
                for obj in to_list(objective_funcs):
                    estimator = clone(self.wf.get_estimator(model, obj))
                    current_estimators.append(estimator)
                    evaluation_model = PoolModel(features=self.X, objective=self.y[obj])
                    evaluation_models.append(evaluation_model)
                
                for acf, alpha_a in zip(to_list(acquisition_function), to_list(alpha)):
                    print('Acquisition function: {}'.format(acf))
                    score_dict[model][acf] = {}

                    for i in range(repeat):
                        random_state_act = random_state+i if random_state != None else None

                        samples, result = run_batch_learning_multi(evaluation_models,
                        aggregation_function=aggregation_function,
                        regression_models=current_estimators,
                        acquisition_function = acquisition_function,
                        pool = np.array(self.active_set_X),
                        batch_size=1,
                        noise=0,
                        initial_samples=initial_samples,
                        active_learning_steps=len(self.active_set_X)-initial_samples,
                        alpha=alpha_a,
                        initialization='random',
                        test_set=np.array(self.test_set_X),
                        random_state=random_state_act,
                        calculate_test_metrics=True,
                        **kwargs)

                        score_dict[model][acf]['iteration {}'.format(i)] = {}
                        score_dict[model][acf]['iteration {}'.format(i)]['samples'] = samples
                        score_dict[model][acf]['iteration {}'.format(i)]['result agg'] = result['aggregated']
                        for j, obj in enumerate(to_list(objective_funcs)):
                            score_dict[model][acf]['iteration {}'.format(i)]['result {}'.format(obj)] = result['model_{}'.format(j)]

            return score_dict
        
    
    def single_step_al():
        pass

    def automatic_al():
        pass

    def update_wf():
        pass
        

