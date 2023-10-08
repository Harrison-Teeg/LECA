########
LECA API
########

.. currentmodule:: LECA

Data Preparation
================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   LECA.prep.arrhenius
   LECA.prep.combine_cut
   LECA.prep.data_visualize
   LECA.prep.direct_sample_arrhenius
   LECA.prep.feature_overview
   LECA.prep.filter_fn
   LECA.prep.import_data
   LECA.prep.interactive_data_visualize
   LECA.prep.outlier_filter

Regression WorkFlow
===================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LECA.fit.WorkFlow

Model Analysis
==============

.. autosummary::
   :toctree: generated/
   :template: function.rst

   LECA.analyze.comparative_datasize_performance
   LECA.analyze.datasize_performance
   LECA.analyze.performance_plot
   LECA.analyze.create_input
   LECA.analyze.predict_conductivity_from_arrhenius_objectives
   LECA.analyze.predict_conductivity_from_log_conductivity_objective
   LECA.analyze.plot_1D
   LECA.analyze.plot_1D_Sx
   LECA.analyze.plot_2D
   LECA.analyze.plot_2D_Sx
   LECA.analyze.predict_arrhenius_fit
   LECA.analyze.visualize_arrhenius_fit
   LECA.analyze.extract_results

LECA Custom Estimators
======================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LECA.estimators.PolynomialRegression
   LECA.estimators.AlphaGPR

Notebook Utils
==============

.. autosummary::
   :toctree: generated/
   :template: function.rst
