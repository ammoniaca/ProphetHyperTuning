import pandas as pd
import numpy as np

from typing import TypeAlias, Literal

Mode_inclusive: TypeAlias = Literal["both", "neither", "left", "right"]


# Mode_axis: TypeAlias = Literal["index", "columns", "rows"]

def get_section_by_datetime(
        df: pd.DataFrame,
        set_days: int,
        from_start: bool = True,
) -> pd.DataFrame:
    df_copy = df.copy(deep=True)

    if from_start:
        first_date_time = min(df_copy['ds'])  # get the "first date" of the dataframe
        cutoff_date = first_date_time + pd.Timedelta(days=set_days)  # cutoff_date = "first date" + days
        section_df = df_copy[df_copy['ds'] <= cutoff_date]
    else:
        end_date_time = max(df_copy['ds'])  # get the "last date" of the dataframe
        cutoff_date = end_date_time - pd.Timedelta(days=set_days)  # cutoff_date = "first date" - days
        section_df = df_copy[df_copy['ds'] >= cutoff_date]
    return section_df


def split_train_test(df: pd.DataFrame, test_days: int):
    df_copy = df.copy(deep=True)
    test = get_section_by_datetime(df=df, set_days=test_days, from_start=False)
    cutoff_date = min(test['ds'])
    train = df_copy[df_copy['ds'] < cutoff_date]
    return train, test


def compute_backward_cap(
        df: pd.DataFrame,
        delta=2
) -> pd.DataFrame:
    """
    Compute CAP (Carrying Capacity). Prophet allows you to make forecasts using a logistic growth trend model,
    with a specified carrying capacity. When forecasting growth, there is usually some maximum achievable
    point: total market size, total population size, etc. This is called the carrying capacity, and the forecast
    should saturate at this point. The important things to note are that cap must be specified for every row in
    the dataframe, and that it does not have to be constant. If the market size is growing, then cap
    can be an increasing sequence.

    :param df:
    :param delta:
    :return:
    """
    df_copy = df.copy(deep=True)
    df_copy['ds'] = pd.to_datetime(df_copy['ds'])
    df_copy['week_of_year'] = df_copy['ds'].dt.isocalendar().week
    df_weekly_max = df_copy.groupby('week_of_year').agg(max=('y', 'max')).reset_index()
    df_weekly_max['max'] = df_weekly_max['max'].apply(np.ceil) + np.ceil(delta)
    df_copy = df_copy.merge(df_weekly_max, how="left")
    df_copy.rename(columns={"max": "cap"}, inplace=True)
    df_copy.drop('week_of_year', axis=1, inplace=True)
    return df_copy


def _increase_change(old, new):
    return (new - old) / abs(old)


def compute_forward_cap(
        past_df: pd.DataFrame,
        future_df: pd.DataFrame,
        delta=2
):
    past_df_copy = past_df.copy(deep=True)
    past_df_copy['ds'] = pd.to_datetime(past_df_copy['ds'])
    past_df_copy.reset_index(inplace=True, drop=True)

    future_df_copy = future_df.copy(deep=True)
    future_df_copy['ds'] = pd.to_datetime(future_df_copy['ds'])
    future_df_copy.reset_index(inplace=True, drop=True)

    # split
    past_df_copy['week_of_year'] = past_df_copy['ds'].dt.isocalendar().week
    # past_df_copy = past_df_copy[(past_df_copy['ds'] <= date_threshold)]
    past_df_weekly_max = past_df_copy.groupby('week_of_year').agg(max=('y', 'max')).reset_index()
    start_value = past_df_weekly_max["max"].iloc[-2]
    end_value = past_df_weekly_max["max"].iloc[-1]
    pc = _increase_change(old=start_value, new=end_value)
    cap_value = (end_value + (end_value * pc)) + delta
    # adding column with constant value
    future_df_copy['cap'] = pd.Series([np.ceil(cap_value) for x in range(len(future_df_copy.index))])
    return future_df_copy
