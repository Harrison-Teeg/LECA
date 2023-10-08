from LECA import prep, fit, analyze    # LECA Modules
import pandas as pd                    # Pandas DataFrames
import numpy as np                     # Numpy for standard math operations mostly
## Imports for interactive widgets
from ipywidgets import interact, widgets, interact_manual
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import itertools
from uncertainties import unumpy, ufloat

#For Custom Kernels:
from sklearn.gaussian_process.kernels import ConstantKernel,RBF,Matern,WhiteKernel,DotProduct,ExpSineSquared,RationalQuadratic



############################## Interactive Notebook Helper Functions ##############################
##
#
def dataset_overview(df, cut_df):
    '''
    Test documentation
    '''
    fts = list(df.columns)
    # Define interactive widgets
    X_slider=widgets.Dropdown(options=fts, description='X')
    Y_slider=widgets.Dropdown(options=fts, description='Y')
    Z_slider=widgets.Dropdown(options=[None]+fts, description='Z (optional)')

    output = widgets.Output()

    # Function call to update plot
    @output.capture()
    def data_vis(change=None):
        clear_output(wait=True)
        display(X_slider)    
        display(Y_slider)    
        display(Z_slider)
        # Make button to plot data
        btn=widgets.Button(description="Run")
        display(btn)
        btn.on_click(data_vis)

        # Initialize plt.Figure object
        with plt.ion():
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot()
            X_feature = X_slider.value
            Y_feature = Y_slider.value
            Z_feature = Z_slider.value
            try:
                if Z_feature:
                    prep.interactive_data_visualize(df, fig, fts, show_axes=[X_feature, Y_feature, Z_feature], highlight_data=cut_df)
                else:
                    prep.interactive_data_visualize(df, fig, fts, show_axes=[X_feature, Y_feature], highlight_data=cut_df)
            except:
                pass
        
        # Show min/max values
        print("\nMin-Max Feature values:")
        for val in fts: print(val, df[val].min(),df[val].max())
    
    data_vis()
    return output
#
#
def interactive_plot(wf, temps, prediction_fn):
    # Initialize plot figure
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)

    # Function to update plot depending on interactive input
    def dynamic_plot(obj, feature_var, fixed_values):
        ax.legend(bbox_to_anchor=(-0.2, 1, 0, 0), loc="upper right", fontsize=12)
        ax.set_ylabel(obj, fontsize=18)
        ax.set_xlabel(feature_var, fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_title(fixed_values, fontsize=12)
        plt.tight_layout()
        plt.show()

    # Copy input features from WorkFlow features (excluding inverse temperature)
    workflow_features = [feature for feature in list(wf.X.columns) if feature != "inverse temperature"]
    varied_feature=widgets.Dropdown(options=workflow_features, description='X')

    fixed_val = {}
    for feature in workflow_features:
        min_val = wf.X_unscaled[feature].min()
        max_val = wf.X_unscaled[feature].max()
        step_val = np.abs(max_val - min_val)/100
        fixed_val[feature] = widgets.FloatSlider(description=feature,min=min_val, max=max_val, step=step_val)

    display(varied_feature)
    for _,widg in fixed_val.items():
        display(widg)


    def update(change=None):
        ax.clear()
        feature_var = varied_feature.value

        for temp in temps:
            arrh_dict = {}
            simulated_values = {}
            for feature, fixed in fixed_val.items():
                simulated_values[feature] = fixed.value

            simulated_values['inverse temperature'] = 1000/(temp+273.15)
            feature_range = list(np.linspace(wf.X_unscaled[feature_var].min(), wf.X_unscaled[feature_var].max(), 30))
            simulated_values[feature_var] = feature_range

            #fill in non-list entries (constants) to match dimensions of our dataset
            for k, v in simulated_values.items():
                if not isinstance(v, list): simulated_values[k] = itertools.repeat(v)
            x_input = pd.DataFrame(zip(*simulated_values.values()), columns = list(simulated_values.keys()))
            fixed_values = ""
            temp_dict = x_input.drop([feature_var],axis=1).iloc[0].to_dict()
            for feature, value in temp_dict.items():
                if feature == "inverse temperature": continue
                fixed_values = fixed_values + feature.replace('x_','') + "=" + str(np.round(value,3)) + ", "
            fixed_values = fixed_values[:-2]

            prediction = prediction_fn(x_input)
            ax.errorbar(feature_range,prediction['conductivity'],yerr=1.96*(prediction['conductivity_std']),xerr=None,label=str(temp)  + r'$\degree$C',alpha=1,capsize=3)
            obj=r'$\sigma$'
        dynamic_plot(obj, feature_var, fixed_values)

    ## Auto - update plot when we change any slider or dropdown menu
    varied_feature.observe(update, 'value')
    for _, fixed in fixed_val.items():
        fixed.observe(update, 'value')

def dataset_composition_overview(df, unit):
    # Look through our col labels for any label with this unit suffix, take the component (and drop suffix)
    components = [component.split(unit,1)[0] for component in df.columns if unit in component]
    component_selection=widgets.Dropdown(options=components, description='Comp:')

            
    #@interact_manual
    def show_compositions(change=None):
        clear_output(wait=True)
        print('Imported DataFrame has shape {}, and columns:\n{}\n'.format(df.shape, list(df.columns)))
        print("Select component, the count of all compositions including that component (>0) will be returned")
        display(component_selection)
        # First slice down the DB to only include datapoints where comp > 0, and only retain columns with component concentrations
        comp = component_selection.value + unit
        loc_df = df.loc[df[comp] > 0.][[element + unit for element in components]]
        
        # Identify unique compositions
        loc_df = loc_df.apply(
        lambda x: ' / '.join(x[x.gt(0)].dropna().index.format()).replace(unit,''), axis=1)
        
        loc_df = loc_df.value_counts(sort=True)
        
        print('Number of unique combinations:',len(loc_df))
        display(loc_df)
        print('Total number of datapoints for compositions: {}'.format(loc_df.sum()))

    ## Auto - run test when we change anything
    component_selection.observe(show_compositions, 'value')
    show_compositions()

class data_generator_1d():
    """
    Interactive synthetic dataset generator.
    Allows for normally distributed systematic/measurment
    error and simulating repeated measurements.
    
    Parameters
    ----------
    X: Array-like
        Set of x values to generate training data for
    
    f: Callable
        Signature: f(x) -> array-like 1d function

    measurement_error: Optional[Float]
        Sigma value for the normally distributed error of each `measurement` of the output of function f(x) -> y.
        Repeated measurements of f(x) will reveal the measurement error.

    systematic_error: Optional[Float]
        Sigma value for the normally distributed error of each `sample` at the point x.
        (This represents a hidden, unchanging deviation from the true function value
        over repeated measurements of f(x))

    n_resample: Optional[Int]
        How many times each x value from ``X`` is measured to generate statistics on the measurement deviations.
    
    Attributes
    ----------
    df: pd.DataFrame
        Dataframe of generated `measurement` values.
    
    """
    def __init__(self, X, f, measurement_error=0.05, systematic_error=0.01, n_resample=5):
        self.df = pd.DataFrame()
        self.f = f
        self.X = pd.DataFrame({'x':X})
        
        mE = widgets.FloatText(        # Measurement noise
        value=measurement_error, 
        min=0,
        description='Meas. Err:',
        disabled=False
    )
        sE = widgets.FloatText(        # Systematic error
        value=systematic_error,
        min=0,
        description='Syst. Err:',
        disabled=False
    )
        n = widgets.BoundedIntText(    # How many times a given x is tested for objective function y (generating statistics)
        value=n_resample,
        min=1,
        max=50,
        description='Repeat:',
        disabled=False
    )

        button = widgets.Button(description = 'Generate')
        ui = widgets.VBox([mE, sE, n, button])
        out = widgets.Output()
        
        def run_experiment(x, measurement_error=measurement_error, systematic_error=systematic_error, n_resample=n_resample): ## take dataframe -> output dataframe of experimental results
            result = pd.DataFrame(columns=['x','y'])
            samples = [ground_truth + np.random.normal(0, systematic_error) for ground_truth in f(x)]
            for _ in range(n_resample):
                measured_y = [y + np.random.normal(0, measurement_error) for y in samples]
                single_sample = pd.concat([x, pd.Series(measured_y, name='y')],axis=1)
                result = pd.concat([result, single_sample],axis=0)
            return result.reset_index(drop=True)
        
        def on_button_clicked(_):
            with out:
                clear_output()

                if n.value > 1:
                    df = prep.combine_cut(run_experiment(self.X, mE.value, sE.value, n.value), 'y', 'x', max_samples=n.value)
                    plt.errorbar(df['x'],df['y'], df['y_std'], fmt='.', capsize=2, label='Training Data')
                else:
                    df = run_experiment(self.X, mE.value, sE.value, n.value)
                    plt.scatter(df['x'],df['y'], label='Training Data')

                x_min, x_max = df['x'].min(), df['x'].max()
                temp_X = np.linspace(x_min, x_max, int(np.abs(x_max-x_min)*10))
                plt.plot(temp_X, f(temp_X), label='True Fn')
                plt.legend()
                plt.show()
                self.df = df
                
            
        button.on_click(on_button_clicked)
        on_button_clicked(None)
        a = widgets.VBox([out, ui])
        display(a)

def prediction_plot_2d(wf, model_name, f): # Todo: generalize this for 2d regression problems (feature/obj::X/y)

    # Initialize plot figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)


    n_samples = 100
    X = np.linspace(-2, 10, n_samples)
    X_df = pd.DataFrame(X, columns=['x'])
    pred = wf.predict(X_df, {'y':model_name}, min_max=True, return_std=True)
    
    ax.fill_between(X, pred['y']-pred['y_std'], pred['y']+pred['y_std'],
                    label=r'$\pm \sigma$', color='orange', alpha=0.5, linestyle='-')
    
    ax.errorbar(wf.X_unscaled['x'],wf.y['y'],yerr=1.96*wf.std['y_std'],xerr=None,label='Training Data', fmt='.', c='black', alpha=0.5,capsize=1)    

    ax.errorbar(X,f(X),label='f(x)', fmt='--', alpha=1)
    ax.errorbar(X,pred['y'],label='Prediction', color='r', alpha=1)

    ax.set_title(model_name)
    ax.set_ylabel('y', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend()
    plt.tight_layout()
    plt.show()
