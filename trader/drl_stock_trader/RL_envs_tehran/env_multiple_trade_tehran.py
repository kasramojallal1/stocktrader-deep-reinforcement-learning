import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100

# initial amount of money we have in our account
with open('trader/drl_stock_trader/config/initial_balance.txt', 'r') as f:
    INITIAL_ACCOUNT_BALANCE = int(f.read())

# total number of stocks in our portfolio
STOCK_DIM = 1

# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

REWARD_SCALING = 1e-4


class StockEnvTradeTehran(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 socket,
                 df,
                 day=0,
                 initial=True,
                 previous_state=[],
                 model_name='',
                 iteration=''):

        self.socket = socket
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))

        # Shape = 7: [Current Balance] + [prices 1] + [owned shares 1] 
        # + [macd 1] + [rsi 1] + [cci 1] + [adx 1]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,))

        # load data from a pandas data frame
        self.data = self.df.loc[self.day, :]

        # if the process if finished of not (at first it is falls, obviously)
        self.terminal = False

        # initialize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [self.data.adjcp] + \
                     [0] + \
                     [self.data.macd] + \
                     [self.data.rsi] + \
                     [self.data.cci] + \
                     [self.data.adx]

        # initialize reward
        self.reward = 0

        # initialize the cost of all transactions
        self.cost = 0

        # number of trades
        self.trades = 0

        # memorize all of the balance changes
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]

        # memorize all of the rewards changes
        self.rewards_memory = []

        # the random seed
        self._seed()

        self.model_name = model_name

        # number of iteration   
        self.iteration = iteration

    # at each step we can either sell, hold or buy (many)
    def _sell_stock(self, action):

        # it will only sell if there is a share available
        if self.state[2] > 0:

            # update balance
            # new balance is the old balance plus (price * number of shares we want * fee)
            self.state[0] += \
                self.state[1] * min(abs(action), self.state[2]) * (1 - TRANSACTION_FEE_PERCENT)

            # update state
            # the specific stocks share value will be updated
            # the min actually shows how many shares are removed (it won't exceed the available amount)
            self.state[2] -= min(abs(action), self.state[2])

            # update the cost
            # the cost will be added by (the price * number of shares we want * fee)
            self.cost += (self.state[1] * min(abs(action), self.state[2]) * TRANSACTION_FEE_PERCENT)[0]

            # number of trades will be added by one
            self.trades += 1

        # there were no shares available for that asset   
        else:
            pass

    # at each step we can either sell, hold or buy (many)
    def _buy_stock(self, action):

        # the available amount shows how many shares of that specific stock can we buy
        # the reason for this is that it is the division of balance by stock prices, so it shows how many can we buy
        available_amount = self.state[0] // self.state[1]

        # update balance
        # new balance is the old balance minus (price * number of shares we want * fee)
        self.state[0] -= self.state[1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

        # update state
        # the specific stocks share value will be updated
        # the min actually shows how many shares are added (it won't exceed the available amount)
        self.state[2] += min(available_amount, action)

        # update the cost
        # the cost will be added by (the price * number of shares we want * fee)
        self.cost += (self.state[1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT)[0]

        # number of trades will be added by one
        self.trades += 1

    # at each step we tell the system to buy, sell or hold number of stocks for each 30 of them
    # [2, 4, 0, 5, -3, 55, -23, ..., 0]    
    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        # check if the process is finished
        if self.terminal:

            # save all of the changes to the balance in a png file
            plt.plot(self.asset_memory, 'r')
            plt.savefig('trader/drl_stock_trader/results_tehran/account_value_trade_{}_{}.png'.format(self.model_name,
                                                                                                      self.iteration))
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)

            print(f'---------------------------------------{self.asset_memory}')
            df_total_value.to_csv(
                'trader/drl_stock_trader/results_tehran/account_value_trade_{}_{}.csv'.format(self.model_name,
                                                                                              self.iteration))

            # all the money = balance + (the balance + (price of assets * number for each one))
            end_total_asset = (self.state[0] + self.state[1] * self.state[2])[0]

            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'previous_total_asset:{self.asset_memory[0]:.2f}'
            }))
            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'end_total_asset:{end_total_asset:.2f}'
            }))
            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'total_reward:{(self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])) - self.asset_memory[0])[0]:.2f}'
            }))
            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'total_cost: {self.cost:.2f}'
            }))
            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'total trades: {self.trades:.2f}'
            }))

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            self.socket.send(text_data=json.dumps({
                'type': 'terminal',
                'message': f'Sharpe: {sharpe:.2f}\n\n'
            }))

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(
                'trader/drl_stock_trader/results_tehran/account_rewards_trade_{}_{}.csv'.format(self.model_name,
                                                                                                self.iteration))

            return self.state, self.reward, self.terminal, {}


        # check if the process is not finished
        else:

            actions = actions * HMAX_NORMALIZE

            # all the money = balance + (the balance + (price of assets * number for each one))
            begin_total_asset = self.state[0] + self.state[1] * self.state[2]

            self._sell_stock(actions)

            self._buy_stock(actions)

            # one day (or a time period) passes
            self.day += 1

            # the new data is from the day till ...
            self.data = self.df.loc[self.day, :]

            # new state parameters are set
            self.state = [self.state[0]] + \
                         [self.data.adjcp] + \
                         [self.state[2]] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]

            # all the money = balance + (the balance + (price of assets * number for each one))
            end_total_asset = (self.state[0] + self.state[1] * self.state[2])[0]

            self.socket.send(text_data=json.dumps({
                'type': 'chart',
                'message': end_total_asset
            }))

            # add this to asset memory
            self.asset_memory.append(end_total_asset)

            ### reward = balance at the end of period - balance at the start of period
            self.reward = end_total_asset - begin_total_asset

            # add the new reward to the reward memory
            self.rewards_memory.append(self.reward)

            # scale the reward
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    # resets everything in the environment and returns the state
    def reset(self):

        # if it is the first iteration
        if self.initial:

            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            # initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                         [self.data.adjcp] + \
                         [0] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]

            # if it is not the first iteration
        else:

            previous_total_asset = self.previous_state[0] + self.previous_state[1] * self.previous_state[2]
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            # initiate state
            self.state = [self.previous_state[0]] + \
                         [self.data.adjcp] + \
                         [self.previous_state[2]] + \
                         [self.data.macd] + \
                         [self.data.rsi] + \
                         [self.data.cci] + \
                         [self.data.adx]

        return self.state

    # returns the state
    def render(self, mode='human', close=False):
        return self.state

    # returns a radom seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
