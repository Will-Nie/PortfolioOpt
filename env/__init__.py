from gym.envs.registration import register

register(id='trading-v0', entry_point='env.gym_env.portfolio_env:StockPortfolioEnvStr1', max_episode_steps=10000)
register(id='trading-v1', entry_point='env.gym_env.portfolio_env:StockPortfolioEnvStr2', max_episode_steps=10000)
