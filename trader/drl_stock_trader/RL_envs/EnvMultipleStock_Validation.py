import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

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
STOCK_DIM = 30

# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
# TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4


class StockEnvValidation(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0, turbulence_threshold=140, iteration=''):
        self.day = day
        self.df = df

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))

        # Shape = 181: [Current Balance] + [prices 1-30] + [owned shares 1-30] 
        # + [macd 1-30] + [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))

        # load data from a pandas data frame
        self.data = self.df.loc[self.day, :]

        # if the process if finished of not (at first it is falls, obviously)
        self.terminal = False

        # sets the threshold for turbulence
        self.turbulence_threshold = turbulence_threshold

        # initialize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()

        # initialize reward
        self.reward = 0

        # initialize turbulence
        self.turbulence = 0

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

        # number of iteration
        self.iteration = iteration

    # at each step we can either sell, hold or buy (many)
    def _sell_stock(self, index, action):

        # if turbulence doesn't exceed the threshold
        if self.turbulence < self.turbulence_threshold:

            # it will only sell if there is a share available
            if self.state[index + STOCK_DIM + 1] > 0:

                # update balance
                # new balance is the old balance plus (price * number of shares we want * fee)
                self.state[0] += \
                    self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * (
                                1 - TRANSACTION_FEE_PERCENT)

                # update state
                # the specific stocks share value will be updated
                # the min actually shows how many shares are removed (it won't exceed the available amount)
                self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])

                # update the cost
                # the cost will be added by (the price * number of shares we want * fee)
                self.cost += self.state[index + 1] * min(abs(action),
                                                         self.state[index + STOCK_DIM + 1]) * TRANSACTION_FEE_PERCENT

                # number of trades will be added by one
                self.trades += 1

            # there were no shares available for that asset    
            else:
                pass


        ##### if turbulence goes over threshold, just clear out all positions    
        else:

            if self.state[index + STOCK_DIM + 1] > 0:

                # update balance
                # new balance is the old balance plus (price * number of shares we want * fee)
                self.state[0] += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * (
                            1 - TRANSACTION_FEE_PERCENT)

                # update state
                # the number for that share will be 0
                self.state[index + STOCK_DIM + 1] = 0

                # update the cost
                # the cost will be added by (the price * number of shares we want * fee)
                self.cost += self.state[index + 1] * self.state[index + STOCK_DIM + 1] * TRANSACTION_FEE_PERCENT

                # number of trades will be added by one
                self.trades += 1

            # there were no shares available for that asset    
            else:
                pass

    # at each step we can either sell, hold or buy (many)
    def _buy_stock(self, index, action):

        # if turbulence doesn't exceed the threshold
        if self.turbulence < self.turbulence_threshold:

            # the available amount shows how many shares of that specific stock can we buy
            # the reason for this is that it is the division of balance by stock prices, so it shows how many can we buy
            available_amount = self.state[0] // self.state[index + 1]

            # update balance
            # new balance is the old balance minus (price * number of shares we want * fee)
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * (1 + TRANSACTION_FEE_PERCENT)

            # update state
            # the specific stocks share value will be updated
            # the min actually shows how many shares are added (it won't exceed the available amount)
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)

            # update the cost
            # the cost will be added by (the price * number of shares we want * fee)
            self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT

            # number of trades will be added by one
            self.trades += 1


        ### if turbulence goes over threshold, just stop buying  
        else:
            pass

    # at each step we tell the system to buy, sell or hold number of stocks for each 30 of them
    # [2, 4, 0, 5, -3, 55, -23, ..., 0]
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        # check if the process is finished
        if self.terminal:

            # save all of the changes to the balance in a png file
            plt.plot(self.asset_memory, 'r')
            plt.savefig('trader/drl_stock_trader/results/account_value_validation_{}.png'.format(self.iteration))
            plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('trader/drl_stock_trader/results/account_value_validation_{}.csv'.format(self.iteration))

            # all the money = balance + (the balance + (price of assets * number for each one))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)

            return self.state, self.reward, self.terminal, {}


        # check if the process is not finished
        else:

            actions = actions * HMAX_NORMALIZE

            ############################# TODO
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            # all the money = balance + (the balance + (price of assets * number for each one))
            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            # it sorts the args from biggest amount of shares to the lowest
            # for instance the 23th arg has the biggest value, so it will be the first arg
            argsort_actions = np.argsort(actions)

            # the actions that have a (-) are sell and the ones with (+) value are buy
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            # first go for the sells
            for index in sell_index:
                self._sell_stock(index, actions[index])

            # second, go for the buys
            for index in buy_index:
                self._buy_stock(index, actions[index])

            # one day (or a time period) passes
            self.day += 1

            # the new data is from the day till ...
            self.data = self.df.loc[self.day, :]

            ## TODO
            self.turbulence = self.data['turbulence'].values[0]

            # new state parameters are set
            self.state = [self.state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]) + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            # all the money = balance + (the balance + (price of assets * number for each one))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

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

        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []

        # initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()

        return self.state

    # returns the state
    def render(self, mode='human', close=False):
        return self.state

    # returns a radom seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
