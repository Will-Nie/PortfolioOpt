import sys
import os
sys.path.append("..")
sys.path.insert(0, os.getcwd())

import gym
import pandas as pd

from env.gym_env.portfolio_env import StockPortfolioEnvStr1
from data.data_demo import data_demo1
from trainer.RLalgo.ppo import main
from trainer.config.ppo_strategy1 import main_config, create_config

# Process your data here [doing data  cleaning, features engineering here]
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
train, test = data_demo1(tech_indicator_list)
stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
feature_dimension = len(tech_indicator_list)
print(f"Feature Dimension: {feature_dimension}")

env_train_kwargs = {
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

env_test_kwargs = {
    'df': test,
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicator_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-1
}

# Choose your algo and train your agent

main('trading-v0', main_config, create_config, env_train_kwargs, env_test_kwargs, 10000)

print('good job')
