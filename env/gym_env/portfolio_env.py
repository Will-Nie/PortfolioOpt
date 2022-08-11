from __future__ import annotations

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import logging
from finquant.portfolio import build_portfolio

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class StockPortfolioEnvStr1(gym.Env):
    """portfolio allocation environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date
    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space, ))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, self.state_space + len(self.tech_indicator_list), self.state_space),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.expand_dims(
            np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            ), 0
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.xlabel('Days')
            plt.ylabel('Cumulative Rewards')
            plt.title('Cumulative Rewards in Days')
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.xlabel('Days')
            plt.ylabel('Rewards')
            plt.title('Rewards in Days')
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = ((252 ** 0.5) * df_daily_return["daily_return"].mean() / df_daily_return["daily_return"].std())
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.expand_dims(
                np.append(
                    np.array(self.covs),
                    [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                    axis=0,
                ), 0
            )
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value / self.initial_amount - 1
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.expand_dims(
            np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            ), 0
        )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({"date": date_list, "daily_return": portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_baseline(self):
        # initalize comparison benchmark
        self.benchmark = build_portfolio(
            names=list(self.data.tic.values),
            start_date=self.df.date[self.df.date.index[0]].unique()[0],
            end_date=self.df.date[self.df.date.index[-1]].unique()[0],
            data_api="yfinance"
        )


class StockPortfolioEnvStr2(gym.Env):
    """portfolio allocation environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date
    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        chosen_stock_num,
        turbulence_threshold=None,
        lookback=252,
        max_step=1000
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.lookback = lookback
        self.df = df
        self.day = np.random.choice(
            np.arange(1, len(self.df.index.unique()))
        )  # randomly initilise a day each time when an env is reset
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space  # stock space is equal to the stock dimension
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.max_step = max_step

        # action_space normalization and shape is self.stock_dim
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_space, ), dtype=np.int)
        self.chosen_stock_num = chosen_stock_num
        #self.action_space = gym.spaces.Box(low=0, high=self.action_space, shape=(self.action_space,), dtype=np.int)
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space * (len(self.tech_indicator_list) + 1), ),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.array([self.data[tech].values.tolist()
                               for tech in self.tech_indicator_list]).flatten().squeeze()
        self.terminal = 0
        self.turbulence_threshold = turbulence_threshold

        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal >= self.max_step
        # print(actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.xlabel('Days')
            plt.ylabel('Cumulative Rewards')
            plt.title('Cumulative Rewards in Days')
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.xlabel('Days')
            plt.ylabel('Rewards')
            plt.title('Rewards in Days')
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = ((252 ** 0.5) * df_daily_return["daily_return"].mean() / df_daily_return["daily_return"].std())
                print("Sharpe per day: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            assert len(np.nonzero(actions)[0]) == self.chosen_stock_num
            weights = np.array([1 / self.chosen_stock_num] * self.chosen_stock_num
                               ).astype(np.float32)  # uniform weights -- can be changed according to strategy
            # print("Normalized actions: ", weights)
            self.actions_memory.append(actions)
            last_day_memory = self.df.loc[self.day - 1, :]

            # load next state
            self.data = self.df.loc[self.day, :]
            self.state = np.array(
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list] + [list(actions)]
            ).flatten().squeeze()
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            chosen_stock = []
            for i in range(actions.shape[0]):
                if actions[i] == 1:
                    chosen_stock.append(self.data.tic.iloc[i])
            portfolio_return = sum(
                (
                    (
                        self.data[self.data.tic.isin(chosen_stock)].close.values /
                        last_day_memory[last_day_memory.tic.isin(chosen_stock)].close.values
                    ) - 1
                ) * weights
            )
            # update portfolio value
            new_portfolio_value = self.initial_amount * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value / self.initial_amount - 1
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling
            self.terminal += 1
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = np.random.choice(np.arange(1, len(self.df.index.unique())))
        self.data = self.df.loc[self.day, :]
        # load states
        if_chosen_feature = np.random.choice([0, 1], size=len(self.data.tic))
        count = 0
        for i in range(len(if_chosen_feature)):
            if if_chosen_feature[i] == 1 and count < self.chosen_stock_num:
                count += 1
            else:
                if_chosen_feature[i] = 0

        self.state = np.array(
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list] + [list(if_chosen_feature)]
        ).flatten().squeeze()
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = 0
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({"date": date_list, "daily_return": portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_baseline(self):
        # initalize comparison benchmark
        self.benchmark = build_portfolio(
            names=list(self.data.tic.values),
            start_date=self.df.date[self.df.date.index[0]].unique()[0],
            end_date=self.df.date[self.df.date.index[-1]].unique()[0],
            data_api="yfinance"
        )
