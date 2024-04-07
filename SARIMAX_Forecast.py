import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product

# The initial parameters are defined

contador = 0
forecast_steps = 4
D = 1
best_aicc = float("inf")
best_params = None
forecast_list_values = []
p_values = range(0, 3)
Q_values = range(0, 2)
P_values = range(0, 2)
s = 7
d = 0
q_values = range(0, 2)
path = 'data.parquet'


def fill_zero_values(df: pd.DataFrame, unique_id: str) -> pd.DataFrame:

    """
    This method fill with zeros the
    dates that aren't present in the max/min value
    of the DataFrame for each unique_id combination.
    """

    unique_id_df = df[df['unique_id'] == unique_id]
    unique_id_df['ds'] = pd.to_datetime(unique_id_df['ds'])
    unique_id_df.set_index('ds', inplace=True)
    data_resampled = unique_id_df.resample('D').asfreq().fillna({'y': 0, 'unique_id': unique_id}).reset_index() # NOQA
    unique_id_df = data_resampled
    return unique_id_df


def obtaining_seasonal_components(df: pd.DataFrame) -> pd.DataFrame:

    """
    It returns the seasonal parameters by
    unique_id by calculating the weight
    of the day of the week in terms of the mean.
    """

    df['day_week'] = df['ds'].dt.dayofweek
    avg_sales_per_day = df.groupby(['unique_id', df['day_week']])['y'].mean().reset_index() # NOQA
    avg_sales_total = df.groupby('unique_id')['y'].mean().reset_index()
    avg_sales_per_day = pd.merge(
        avg_sales_per_day,
        avg_sales_total,
        on='unique_id',
        suffixes=('_day', '_total'),
    )
    avg_sales_per_day['seasonality'] = avg_sales_per_day['y_day'] / avg_sales_per_day['y_total'] # NOQA
    return avg_sales_per_day


def getting_arima_d_parameter(df: pd.DataFrame, d: int) -> int:

    """
    This method determines the optimal value of the differencing
    parameter of the ARIMA model.
    """

    try:
        adftest = adfuller(df['y'], autolag='AIC')
        if adftest[1] > 0.05:
            d = 1
    except:
        d = 1
    return d


def building_arima_model(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        unique_id: str,
        p: int,
        d: int,
        q: int,
        P: int,
        Q: int,
        s: int,
        seasonality: pd.DataFrame,
) -> None:

    """
    This function builds the ARIMA model with the train_data, and also
    appends the results to a forecast_list_values list based on the parameters
    (p,d,q).
    """

    try:
        model = SARIMAX(
            df_train['y'],
            order=(p, d, q),
            seasonal_order=(P, 0, Q, s),
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(
            steps=forecast_steps,
            dynamic=False,
        )
        last_values = df_test['y'][:forecast_steps].values
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(
                start=df_train['ds'].max() + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq='D'
            ),
            'unique_id': unique_id,
            'product_description': df_test['product_description'][:forecast_steps].values,
            'y_orig': last_values,
            'y_pred': forecast,
        })
        forecast_df['day_week'] = forecast_df['ds'].dt.dayofweek
        merged_df = forecast_df.merge(
            seasonality,
            on=[
                'unique_id',
                'day_week'
            ],
            how='inner'
        )
        merged_df['y_pred_adj'] = merged_df['y_pred'] * merged_df['seasonality']
        merged_df['y_pred_adj_rounded'] = merged_df['y_pred_adj'].round(0)
        forecast_list_values.append(merged_df)
        print(
            'Successful iteration, forecast progress in %:',
            (contador/len(unique_ids))*100)
    except:
        print(
            'Failed iteration, forecast progress in %:',
            (contador/len(unique_ids))*100)


def best_arima_parameters(
    param_comb: list[tuple],
    d: int,
    best_aicc: float,
    best_params: tuple,
    train_data: pd.DataFrame,
) -> tuple:

    """
    Method that calculates the best p, q parameters
    for the ARIMA Model, for that unique_id combination
    """

    for params in param_comb:
        p, d, q = params[0], d, params[1]
        try:
            model = ARIMA(train_data['y'], order=(p, d, q))
            model_fit = model.fit(method='statespace')
            aicc = model_fit.aicc
            if aicc < best_aicc:
                best_aicc = aicc
                best_params = params
        except:
            if best_params is None:
                best_params = (0, 0)
    return best_params


def best_sarimax_parameters(
        param_comb: list[tuple],
        d: int,
        D: int,
        s: int,
        best_aicc: float,
        best_params: tuple,
        train_data: pd.DataFrame,
) -> tuple:

    for params in param_comb:
        p, q, P, Q = params[0], params[1], params[2], params[3]
        try:
            model = SARIMAX(
                train_data['y'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
            )
            model_fit = model.fit()
            aicc = model_fit.aicc
            if aicc < best_aicc:
                best_aicc = aicc
                best_params = params
        except:
            if best_params is None:
                best_params = (1, 1, 1, 1)
    return best_params


def deleting_combinations_without_sales(df: pd.DataFrame) -> pd.DataFrame:

    total_sales = df.groupby('unique_id')['y'].sum()
    filtered_unique_ids = total_sales[total_sales != 0].index
    df = df[df['unique_id'].isin(filtered_unique_ids)]
    return df


def cleaning_and_formatting_the_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df['unique_id'] = df['cadena'] + '#' + df['codigo_local_cadena'] + '#' + df['sku_cadena'] # NOQA
    df.rename(
        columns={
            'fecha': 'ds',
            'venta_unidades_dia': 'y',
            'descripcion': 'product_description',
        },
        inplace=True,
    )
    df = df.filter(items=['ds', 'unique_id', 'product_description', 'y'])
    df['ds'] = pd.to_datetime(df['ds'])
    df.drop_duplicates(subset=['unique_id', 'ds'], inplace=True)
    return df


def filtering_last_n_days(df: pd.DataFrame, n: int) -> pd.DataFrame:

    df = df[df['ds'] >= df['ds'].max()-pd.Timedelta(days=n)]
    return df

data_parquet = pd.read_parquet(path)
data_parquet = cleaning_and_formatting_the_dataframe(data_parquet)
data_parquet = filtering_last_n_days(data_parquet, 35)
data_parquet = deleting_combinations_without_sales(data_parquet)

# TESTING
unique_ids_to_filter = [
   'WALMART#511#275040',
   'WALMART#54#274664',
   'WALMART#553#275109',
   'WALMART#610#342588',
   'WALMART#959#425329',
]
filtered_df = data_parquet[
    data_parquet['unique_id'].isin(unique_ids_to_filter)
]
data_parquet = filtered_df
# TESTING

unique_ids = data_parquet['unique_id'].unique()

for unique_id in unique_ids:

    unique_id_df = fill_zero_values(data_parquet, unique_id)
    train_size = int(0.8 * len(unique_id_df))
    train_data, test_data = unique_id_df[:train_size], unique_id_df[train_size:]
    seasonal_components = obtaining_seasonal_components(train_data)
    d = getting_arima_d_parameter(train_data, d)
    # param_combinations = list(product(p_values, q_values))
    param_comb = list(product(
        p_values,
        q_values,
        P_values,
        Q_values,
        )
    )
    best_params = best_sarimax_parameters(
        param_comb,
        d,
        D,
        s,
        best_aicc,
        best_params,
        train_data,
    )
    """ best_params = best_arima_parameters(
        param_combinations,
        d,
        best_aicc,
        best_params,
        train_data,
    ) """

    # p, q = best_params
    p, q, P, Q = best_params
    contador += 1
    building_arima_model(
        train_data,
        test_data,
        unique_id,
        p,
        d,
        q,
        P,
        Q,
        s,
        seasonal_components
    )
print(forecast_list_values)
combined_forecast_list_values = pd.concat(
    forecast_list_values,
    ignore_index=True
)
combined_forecast_list_values.to_csv('testing_arima_final_v3.csv')
