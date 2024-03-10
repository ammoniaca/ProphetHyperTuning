"""

https://facebook.github.io/prophet/docs/diagnostics.html

Parameters that can be tuned

- changepoint_prior_scale: indicates how flexible the changepoints are allowed to be. In other words,
    how much can the changepoints fit to the data. If you make it high it will be more flexible, but you can end up
    overfitting. By default, this parameter is set to 0.05. a range of [0.001, 0.5] would likely be about right.
    Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.

- seasonality_prior_scale: This parameter controls the flexibility of the seasonality. Similarly, a large value
    allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality.
    The default is 10., which applies basically no regularization. That is because we very rarely see overfitting
    here (there’s inherent regularization with the fact that it is being modeled with a truncated Fourier series,
    so it’s essentially low-pass filtered). A reasonable range for tuning it would probably be [0.01, 10];
    when set to 0.01 you should find that the magnitude of seasonality is forced to be very small.

- holiday_prior_scale:  it is used to smoothning the effect of holidays. By default, its value is 10, which
    provides very little regularization. Reducing this parameter dampens holiday effects. This could also be tuned
    on a range of [0.01, 10] as with seasonality_prior_scale.

- seasonality_mode: there are 2 types model seasonality mode, 'additive' (default) OR 'multiplicative'.  Default is
    'additive', but many business time series will have multiplicative seasonality. This is best identified just
    from looking at the time series and seeing if the magnitude of seasonal fluctuations grows with the magnitude
    of the time series (see the documentation here on multiplicative seasonality), but when that isn’t possible,
    it could be tuned.


Maybe tune?

- changepoint_range: This is the proportion of the history in which the trend is allowed to change. This defaults
    to 0.8, 80% of the history, meaning the model will not fit any trend changes in the last 20% of the time series.
    In a fully-automated setting, it may be beneficial to be less conservative. In that setting,
    [0.8, 0.95] may be a reasonable range.


Parameters that would likely not be tuned

- growth: Options are ‘linear’ and ‘logistic’. This likely will not be tuned; if there is a known saturating point
    and growth towards that point it will be included and the logistic trend will be used, otherwise it will be linear.

- n_changepoints: number of change happen in the data. Prophet model detects them by its own. By default, its value
    is 25, which are uniformly placed in the first 80% of the time series. Changing n_changepoints can add
    value to the model. IMPORTANT: Rather than increasing or decreasing the number of changepoints, it will likely
    be more effective to focus on increasing or decreasing the flexibility at those trend changes,
    which is done with changepoint_prior_scale.

- yearly_seasonality: By default (‘auto’) this will turn yearly seasonality on if there is a year of data, and off
    otherwise. Options are [‘auto’, True, False]. If there is more than a year of data, rather than trying to turn
    this off during HPO, it will likely be more effective to leave it on and turn down seasonal effects
    by tuning seasonality_prior_scale.

- weekly_seasonality: Same as for yearly_seasonality, i.e., [‘auto’, True, False].

- daily_seasonality: Same as for yearly_seasonality, i.e., [‘auto’, True, False].


# TODO
seasonalities: seasonalities with fourier_order Prophet model, by default finds the seasonalities and adds the
    default parameters of the seasonality. We can modify the seasonalities effect by adding custom seasonalities as
    add_seasonality in the model with different fourier order. By default, Prophet uses a Fourier order of 3 for
    weekly seasonality and 10 for yearly seasonality.



Example

param_grid = {
    'changepoint_prior_scale': [i/1000 for i in range(1, 501, 1)],
    'seasonality_prior_scale': [i/100 for i in range(1, 1001, 1)],
    # 'holiday_prior_scale': [i/100 for i in range(1, 1001, 1)],
    'seasonality_mode': ['additive', 'multiplicative'], # see 'additive' is good for our dataset
    'changepoint_range': [i/100 for i in range(80, 100, 1)],
    'growth': ['linear', 'logistic'], # see 'logistic' is good for our dataset
    'n_changepoints': [i for i in range(1, 100, 1)] # ATTENTION: can be avoided (see doc)
    'yearly_seasonality': ['auto', True, False],
    'weekly_seasonality': ['auto', True, False],
    'daily_seasonality': ['auto', True, False]
}
:param verbose: bool, default: False (disable verbose output)

"""
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict

from prophet import Prophet
from prophet_hypertuning.utils import split_train_test, compute_backward_cap, compute_forward_cap


def prophet_hyper_tuning(
        df: pd.DataFrame,
        params: dict,
        test_days: int,
        evaluation_metric,
        set_delta: int = 2,
        logy: bool = True,
        save: bool = False
) -> tuple:
    """

    :param df: pd.DataFrame
    :param test_days:
    :param params:
    :param evaluation_metric:
    :param set_delta: [0, +inf]
    :param logy: if TRUE convert 'y' in natural logarithm before training the model
    :param save: bool, default: False (disable save output)
    :return:
    """
    # TODO: check if ds and y are present
    df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])
    set_delta = np.abs(set_delta)

    # Create Train and Test dataframe
    train, test = split_train_test(df, test_days=test_days)
    # if loggy: TRUE, convert to natural logarithm
    if logy:
        train.loc[:, "y"] = np.log(df.loc[:, 'y'])
        # log(0) is not defined!!!
        if set_delta > 0:
            set_delta = np.log(set_delta)

    # create future dataframe for prediction
    future = test.drop(columns=['y'])

    # Compute CAP in Train and in future (only for "growth: logistic")
    train = compute_backward_cap(train, delta=set_delta)
    future = compute_forward_cap(past_df=train, future_df=future, delta=set_delta)

    # Defining a dict
    tuning_results = defaultdict(list)
    # Generate all combinations of parameters
    keys, values = zip(*params.items())
    best_score = np.inf
    best_params = {}
    index = 1
    for v in itertools.product(*values):
        experiment = dict(zip(keys, v))
        print(experiment)
        _add(tuning_results, experiment)
        model = Prophet(**experiment).fit(train)  # Fit model with given params
        forecast = model.predict(future)
        # if loggy: TRUE, convert to exponential
        if logy:
            _feature_name = list(forecast.columns)
            _feature_name.remove('ds')
            forecast.loc[:, _feature_name] = np.exp(forecast.loc[:, _feature_name])
        result = pd.merge(forecast, test, on="ds", how="left")
        if save:
            result.to_csv(f"{index}.csv", sep=";", index_label=False)
            tuning_results['file'].append(f"{index}.csv")
            index += 1
        score = evaluation_metric(list(result['y']), list(result['yhat']))
        if score < best_score:
            best_score = score
            best_params = experiment
            # best_params['best_score'] = best_score
        tuning_results['score'].append(score)
    tuning_results_df = pd.DataFrame.from_dict(tuning_results)
    return best_score, best_params, tuning_results_df


def _add(dict1: dict, dict2: dict):
    for key, value in dict2.items():
        dict1[key].append(value)
