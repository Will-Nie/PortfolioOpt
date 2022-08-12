# This project is for portfolio optimisation with RL (Of course can be generalised into other methods - just add your strategy in trainer and execute in main)

# The repo is structured as follows:

1. ``data`` file  represents how data is processed. is one wants to add different ways of analysing and processing data, please write a function there.

2. ``env`` file includes basic env for portfolio optimisation with RL. If ones use their own data, please make sure ``close`` price for each datum is appended at least. Feature change can be adapted by changing ``self.observation_space``.

3. ``main`` file includes the entry for different strategies. By strategies, we mean a specific way of processing data + a env design

4. ``model`` the model the agent learns

5. ``results`` the return and cumulative returns for the agent in the test set

6.  ``trainer`` This part is comprised of two parts. In principle, ``RLalgo`` should not be modified unless ones wants to add more algorithms. ``config`` can be modified as wishes.

7.  ``utils`` includes all necessities supporting the agent to run.

# To run

Please directly run the strategy in the ``main`` file
