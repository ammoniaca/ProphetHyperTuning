"""
- n_changepoints: number of change happen in the data. Prophet model detects them by its own. By default, its value
    is 25, which are uniformly placed in the first 80% of the time series. Changing n_changepoints can add
    value to the model.

- changepoint_prior_scale: indicates how flexible the changepoints are allowed to be. In other words,
    how much can the changepoints fit to the data. If you make it high it will be more flexible, but you can end up
    overfitting. By default, this parameter is set to 0.05.

- seasonality_mode: there are 2 types model seasonality mode. Additive and multiplicaticative. By default, Prophet
    fits additive seasonalities, meaning the effect of the seasonality is added to the trend to get the forecast.
    Prophet can model multiplicative seasonality by setting seasonality_mode='multiplicative' in the model.

- holiday_prior_scale:  it is used to smoothning the effect of holidays. By default, its value is 10, which
    provides very little regularization. Reducing this parameter dampens holiday effects.

- seasonalities: seasonalities with fourier_order Prophet model, by default finds the seasonalities and adds the
    default parameters of the seasonality. We can modify the seasonalities effect by adding custom seasonalities as
    add_seasonality in the model with different fourier order.Yy default Prophet uses a Fourier order of 3 for
    weekly seasonality and 10 for yearly seasonality.

param_grid = {
    'n_changepoints': [i for i in range(1, 100, 1)]
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

"""



from prophet import Prophet
import pandas as pd
import itertools


def phyto(
        data: pd.DataFrame,
        days: int,
        params: dict,
        score_metric,
        verbose: bool = False
) -> dict:
    """
    :param data:
    :param days:
    :param params:
    :param score_metric:
    :param verbose: bool, default: False (disable verbose output)
    :return:
    """
    # Generate all combinations of parameters


    pass
