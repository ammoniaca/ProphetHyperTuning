import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error


def _mape(actual, pred):
    """
    Mean Absolute Percentage Error (MAPE) Function

    input: list/series for actual values and predicted values
    output: mape value
    """
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def _wmape(y_true, y_pred):
    """
    This is an adapted version of MAPE (Mean Absolute Percentage Error) that solves
    the problem of division by zero when there are no sales for a specific day.

    output: modified mape
    """
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def median_absolute_percentage_error(actual, predicted):
    """
    Median Absolute Percentage Error (MDAPE) is an error metric used to measure the
    performance of regression machine learning models. It is the median of all absolute
    percentage errors calculated between the predictions and their corresponding actual values.

    https://stephenallwright.com/mdape/

    <10%        Very good
    10%-20%	    Good
    20%-50%	    OK
    >50%	    Not good

    :param actual:
    :param predicted:
    :return:
    """
    return np.median((np.abs(np.subtract(actual, predicted) / actual))) * 100


def symmetric_mean_absolute_percentage_error(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)

    :param actual:
    :param predicted:
    :return:
    """
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual))/2)) * 100

    # return 100 / len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def model_evaluation(actual, predicted):
    """
    Evaluation model result with:

    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - MDAPE (Mean Absolute Percentage Error)
    - SMAPE (Symmetric Mean Absolute Percentage Error)

    :param actual:
    :param predicted:
    :return: dict
    """
    eval_model = {
        # Mean Squared Error (MSE)
        'mse': mean_squared_error(actual, predicted),
        # Root mean squared error (RMSE)
        'rmse': root_mean_squared_error(actual, predicted),
        # Mean Absolute Error (MAE)
        'mae': mean_absolute_error(actual, predicted),
        # Mean absolute percentage error (MAPE)
        'mape': mean_absolute_percentage_error(actual, predicted),
        # Meadin Absolute Percentage Error (MDAPE)
        'mdape': median_absolute_percentage_error(actual, predicted),
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        'smape': symmetric_mean_absolute_percentage_error(actual, predicted)
    }
    return eval_model
