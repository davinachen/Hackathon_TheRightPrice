# Model

Our team developed two Notebooks to address the following, and finally opted for an XGBoost model that we tuned with Bayesian Optimization, as well as the linear projections through time horizons.
   
1. Data Exploration and Visualization
We ran the following analyses on some features to get a grasp of the dataset : 
  - Summary Statistics by feature
  - Heatmap of Correlation matrix
  - Statistical Distribution of Target Variable
  - QQ-plots
  - Time-Series decomposition
  - Box-plot
  - Dickey-Fuller Hypothesis Testing for stationarity

2. Models Training
We trained many models assessed on many metrics (RMSE, MAE, R², MAPE,etc.):
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Robust (huber) Regression
  - Random Forest
  - XGBoost ---------------------> Final choice. Model Tuned with Bayesian Optimization (hyperopt)
  - Gradient Boosting (sklearn)

3. Others
On top of 1 & 2, we made a few attempts at stacking, but it is not documented on the notebook.

Later, We analyzed the 'valeurfonc' as time-series :
   - Decomposition
   - Removal of Seasonal Component
   - ACF and PACF on residual ------------> Did not indicate autocorrelation. So no ARMA model was implemented.

Instead, we created 1 model per department (to account for the differences of dynamics), composed of :
  - Seasonal component (monthly seasonal means)
  - Linear trend coefficient (slope)

We used the slope to project the prices along time horizons in the demo tools.

