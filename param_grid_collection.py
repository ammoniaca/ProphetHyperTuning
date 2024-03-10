

param_grid_example = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.01, 0.1, 1],
    'seasonality_mode': ['additive'],  # see 'additive' is good for our dataset
    'changepoint_range': [0.85, 0.95, 0.99],
    'growth': ['logistic'],  # see 'logistic' is good for our dataset
    'n_changepoints': [10, 25, 50],  # ATTENTION: can be avoided (see doc)
    'yearly_seasonality': ['auto'],
    'weekly_seasonality': ['auto'],
    'daily_seasonality': ['auto']
}


param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1, 5, 10],
    'seasonality_mode': ['additive', 'multiplicative'],  # see 'additive' is good for our dataset
    'changepoint_range': [0.75, 0.8, 0.85, 0.95, 0.99],
    'growth': ['linear', 'logistic'],  # see 'logistic' is good for our dataset
    'n_changepoints': [10, 25, 50, 75, 100, 150, 200],  # ATTENTION: can be avoided (see doc)
    'yearly_seasonality': ['auto', True, False],
    'weekly_seasonality': ['auto', True, False],
    'daily_seasonality': ['auto', True, False]
}