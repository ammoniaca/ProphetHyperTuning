"""
use the command pip install -r requirements. txt in your terminal
"""
from prophet_hypertuning.phypertuning import prophet_hyper_tuning
from prophet_hypertuning.utils import get_section_by_datetime
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error as mape

from param_grid_collection import param_grid, param_grid_example

if __name__ == '__main__':
    # EXAMPLE
    # Data Preparation
    df_original = pd.read_csv('Poplar_casale_clean.csv', sep=",")
    df = df_original.copy(deep=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # Get dvina_irr_4 sensor data
    dvina_irr_4_df = df[['ds', 'Dvina_irr_4']].copy(deep=True)
    dvina_irr_4_df.rename(columns={"Dvina_irr_4": "y"}, inplace=True)

    # CREATE DATAFRAME FOR HYPERMATERMATER TUNING (e.g., the first month of data)
    df = get_section_by_datetime(df=dvina_irr_4_df, set_days=30, from_start=True)

    # SET paramter grid for Hyperparamters tuning of Prophet

    best_score, best_params, tuning_results_df = prophet_hyper_tuning(
        df=df,
        params=param_grid_example,
        test_days=3,
        evaluation_metric=mape
    )
    print("")
    tuning_results_df.to_csv("tuning_result.csv", sep=';', index_label=False)
