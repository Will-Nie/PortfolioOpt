from finrl import config_tickers
from data.utils import YahooDownloader, FeatureEngineer, data_split
import pandas as pd


def data_demo1(tech_indicator_list):
    df = YahooDownloader(
        start_date='2008-01-01', end_date='2021-09-02', ticker_list=config_tickers.DOW_30_TICKER
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicator_list,
        use_turbulence=False,
        user_defined_feature=False
    )

    df = fe.preprocess_data(df)

    # add covariance matrix as states
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    train = data_split(df, '2009-01-01', '2018-06-30')

    test = data_split(df, '2018-07-15', '2020-06-30')

    return train, test
