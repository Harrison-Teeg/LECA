LECA - The Liquid Electrolyte Composition Analysis Package
==========================================================

The Liquid Electrolyte Composition Analysis (LECA) package creates
a simplified Jupyter-Notebook [1]_ based workflow for applying
Scikit-Learn [2]_ machine learning regression models to predict 
liquid electrolyte behavior based on composition.

Requirements
============

Python 3.7+, Jupyter Notebook 6.4.11+

With the following python libraries:
        - Scikit-learn 1.3.1+
        - NumPy 1.22.3+
        - Matplotlib 3.5.1+
        - Pandas 1.4.2+
        - SciPy 1.8.1+
        - Uncertainties 3.1.7+
        - MAPIE 0.5.6
        - HDBSCAN 0.8.28+
        - Seaborn 0.11.2+
        - GPyOpt 1.2.6+

Installation
============

The LECA package dependencies can be installed from the LECA package base directory with:

.. code-block:: bash

   pip install -r requirements.txt

The following miniconda-installation method is known to work:

.. code-block:: bash

   # Run these commands in LECA base directory
   conda create -n leca-env python=3.8
   conda activate leca-env
   pip install -r requirements.txt

From this directory simply run a jupyter notebook and import the LECA modules with:

.. code:: python

   from LECA import prep, fit, analyze


Envisaged LECA Work Flow
========================

.. image:: ../../source/images/LECA_overview.png
   :width: 600
   :align: center

Data Import and Feature Engineering
-----------------------------------
        - Import Data from JSON or CSV files
        - Visualize dataset with feature overview and interactive data visualizer
        - Identify and filter outlier values using HDBSCAN [3]_
        - Manually filter nonsense-values with user defined explicit boundaries
        - Combine repeated measurements and record statistical behavior (measurement noise)
        - Generate surrogate models for Arrhenius behavior or other user defined values

Initialize Regression Models and Compare Results
------------------------------------------------
        - Data splitting / Scaling automatically handled
        - Declare regression models to implement (supports N-dimensional feature/objective space)
                - Linear Regression (LR)
                - Gaussian Process Regression (GPR) (supports Isotropic/Anisotropic RBF, Matern, RQ, Custom kernel)
                - Neural Network (NN)
                - Random Forest (RF)
        - Hyperparameter Optimization for NN and RF with GPyOpt [4]_
        - Customized Polynomial selection for LR [5]_
        - Cross-validated scoring of models and visualization to provide simple overview of comparative model performance
        - Ensemble based uncertainty estimation for LR / NN / RF models using MAPIE [6]_
        - Validate performance of models on unseen validation data


Analyze Objective Function for Compositions
-------------------------------------------
        - Interactive widgets to visualize objective function and model uncertainty for various compositions
        - Return optimal composition to maximize/minimize objective function optimization
        - Ranked Batch Mode Active Learning module based on RBMAL approach of Cordoso et al. [7]_

Areas of Further Development
============================


Multi-Objective Optimization: Identifying Pareto-fronts for multiple-objectives for electrolyte composition (e.g. electrochemical stability, conductivity, etc.)

References
==========

.. [1] [9] Brian E. Granger and Fernando Pérez. “Jupyter: Thinking and Storytelling With Code and Data”. In: Computing in Science & Engineering 23.2 (2021), pp. 7–14. doi: 10.1109/MCSE.2021.3059263.

.. [2] F. Pedregosa et al. “Scikit-learn: Machine Learning in Python”. In: Journal of Machine Learning Research 12 (2011), pp. 2825–2830.

.. [3] Leland McInnes, John Healy, and Steve Astels. “hdbscan: Hierarchical density based clustering”. In: The Journal of Open Source Software 2.11 (2017), p. 205.

.. [4] The GPyOpt authors. GPyOpt: A Bayesian Optimization framework in python. 2016. url: http://github.com/SheffieldML/GPyOpt.

.. [5] Anand Narayanan Krishnamoorthy et al. “Data-Driven Analysis of High-Throughput Experiments on Liquid Battery Electrolyte Formulations: Unraveling the Impact of Composition on Conductivity**”. In: Chemistry–Methods 2.9 (2022), e202200008. doi: https://doi.org/10.1002/cmtd.202200008. 

.. [6] MAPIE - Model Agnostic Prediction Interval Estimator. Version: 0.4.1. url: https://mapie.readthedocs.io/en/latest/index.html (visited on 08/24/2022).
 
.. [7] Thiago N.C. Cardoso et al. "Ranked batch-mode active learning". In: Information Sciences 379 (2017) pp. 313-337. doi: https://doi.org/10.1016/j.ins.2016.10.037

Acknowledgments
===============

This project has received funding from the European Union’s Horizon 2020 research and innovation program under grants agreement No 957189 (BIG-MAP) and No 957213 (BATTERY2030+). 
