import gym
import pandas as pd
from env.gym_env.portfolio_env import StockPortfolioEnvStr1
from finquant.portfolio import build_portfolio
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers

names = ['GOOG', 'AMZN', 'MCD', 'DIS']
start_date = '2015-01-01'
end_date = '2017-12-31'
df = build_portfolio(names=names, start_date=start_date, end_date=end_date,  data_api="yfinance")

df = YahooDownloader(start_date = '2008-01-01',
                     end_date = '2021-09-02',
                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
print('Done')
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

df = fe.preprocess_data(df)

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values 
  cov_list.append(covs)

  
df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)

train = data_split(df, '2009-01-01','2020-06-30')



stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
feature_dimension = len(tech_indicator_list)
print(f"Feature Dimension: {feature_dimension}")




env_kwargs = {
    'df': train,
    "hmax": 100, 
    "initial_amount": 1000000, 
    "transaction_cost_pct": 0, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": tech_indicator_list, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-1
    
}

env = gym.make('trading-v0', **env_kwargs)

print('good job')



# below is the code snippet for programming pointer network
'''
play with the env if interested
env = gym.make('trading-v1', **env_train_kwargs)
env.reset()
env.step(np.array([0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1]))
'''


from easydict import EasyDict
import torch
from model.trading_pointer import PointerNetwork
embedding = torch.rand(size=(1, 140,))
entity_embedings = torch.rand(size=(1, 28, 16,))
entity_mask = torch.rand(size=(1, 28,)) > 0

default_model_config = EasyDict({
    'model': {'input_dim': 140, 'max_selected_units_num': 14, 'max_entity_num': 28,
                'entity_embedding_dim': 16, 'key_dim': 32, 'func_dim': 256,
                'lstm_hidden_dim': 32, 'lstm_num_layers': 1,
                'activation': 'relu', 'entity_reduce_type': 'selected_units_num',# ['constant', 'entity_num', 'selected_units_num']
                }
}
)
net = PointerNetwork(default_model_config)
logits1, units, embedding1 = net.forward(embedding, entity_embedings, entity_mask, temperature=0.8) # trainer, entity mask --> boolean