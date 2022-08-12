import sys
import os
sys.path.append("..")
sys.path.insert(0, os.getcwd())

import gym
import pandas as pd
import numpy as np

from env.gym_env.portfolio_env import StockPortfolioEnvStr1
from data.data_demo import data_demo1
from trainer.RLalgo.impala import ReinforcementLoss, FeaturesSerial, trainer

# Process your data here [doing data  cleaning, features engineering here]
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
train, test = data_demo1(tech_indicator_list)
chosen_stock_number = 14
max_step = 1000
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
    "reward_scaling": 1e-1,
    "chosen_stock_num": chosen_stock_number,
    "max_step": max_step
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
    "reward_scaling": 1e-1,
    "chosen_stock_num": chosen_stock_number,
    "max_step": max_step
}

model_config = {
    'model': {
        'input_dim': 140,
        'max_selected_units_num': chosen_stock_number,
        'max_entity_num': stock_dimension,
        'entity_embedding_dim': 16,
        'key_dim': 32,
        'func_dim': 256,
        'lstm_hidden_dim': 32,
        'lstm_num_layers': 1,
        'activation': 'relu',
        'entity_reduce_type': 'selected_units_num',  # ['constant', 'entity_num', 'selected_units_num']
    }
}

optimiser_config = {
    'cfg': {
        'type': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0,
        'eps': 1e-8,
        'decay': 0.009,
        'momentum': 0.9,
        'amsgrad': False
    }
}

# Choose your algo and train your agent
trainer(FeaturesSerial, ReinforcementLoss, 100, 'trading-v1', env_train_kwargs, model_config, optimiser_config)

print('good job')
