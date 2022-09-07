import time
import numpy as np
import pandas as pd
import json

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO

from trader.drl_stock_trader.RL_envs.EnvMultipleStock_Trade import StockEnvTrade
from trader.drl_stock_trader.RL_envs_tehran.env_multiple_trade_tehran import StockEnvTradeTehran

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from trader.drl_stock_trader.config import config
from trader.drl_stock_trader.preprocess import data_split


# Advantage Actor Critic
def train_A2C(socket, env_train, model_name, timesteps, market):
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    if market == 'dow':

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (A2C): {(end - start) / 60:.2f} minutes'
        }))
        return model

    else:

        model.save(f"{config.TRAINED_MODEL_DIR_TEHRAN}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (A2C): {(end - start) / 60:.2f} minutes'
        }))
        return model


# Proximal Policy Optimization
def train_PPO(socket, env_train, model_name, timesteps, market):
    start = time.time()
    model = PPO('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    if market == 'dow':

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (PPO): {(end - start) / 60:.2f} minutes'
        }))
        return model

    else:

        model.save(f"{config.TRAINED_MODEL_DIR_TEHRAN}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (PPO): {(end - start) / 60:.2f} minutes'
        }))
        return model


# Deep Deterministic Policy Gradient
def train_DDPG(socket, env_train, model_name, timesteps, market):
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    if market == 'dow':

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (DDPG): {(end - start) / 60:.2f} minutes'
        }))
        return model

    else:

        model.save(f"{config.TRAINED_MODEL_DIR_TEHRAN}/{model_name}")
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'Training time (DDPG): {(end - start) / 60:.2f} minutes'
        }))
        return model


# validation of algorithms
def algorithms_validation(model, test_data, test_environment, test_observation) -> None:
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_observation)
        test_observation, rewards, done, info = test_environment.step(action)


# get the sharp ratio of algorithms      
def sharpe_ratio_of_algorithms(iteration, market):
    if market == 'dow':

        df_total_value = pd.read_csv('trader/drl_stock_trader/results/account_value_validation_{}.csv'.format(iteration), index_col=0)
        df_total_value.columns = ['account_value_train']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()

        return sharpe

    else:

        df_total_value = pd.read_csv('trader/drl_stock_trader/results_tehran/account_value_validation_{}.csv'.format(iteration), index_col=0)
        df_total_value.columns = ['account_value_train']
        df_total_value['daily_return'] = df_total_value.pct_change(1)
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()

        return sharpe


### make a prediction based on trained model ###
def DRL_prediction(socket,
                   df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   initial):
    ## trading environment
    trade_data = data_split(df,
                            start_date=unique_trade_date[iter_num - rebalance_window],
                            end_date=unique_trade_date[iter_num])

    trading_environment = DummyVecEnv([lambda: StockEnvTrade(socket=socket,
                                                             df=trade_data,
                                                             initial=initial,
                                                             previous_state=last_state,
                                                             model_name=name,
                                                             iteration=iter_num)])
    trading_observation = trading_environment.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(trading_observation)
        trading_observation, rewards, dones, info = trading_environment.step(action)

        if i == (len(trade_data.index.unique()) - 2):
            last_state = trading_environment.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('trader/drl_stock_trader/results/last_state_{}_{}.csv'.format(name, i), index=False)

    return last_state


def DRL_prediction_Tehran(socket,
                          df,
                          model,
                          name,
                          last_state,
                          iter_num,
                          unique_trade_date,
                          rebalance_window,
                          initial):
    ## trading environment
    trade_data = data_split(df,
                            start_date=unique_trade_date[iter_num - rebalance_window],
                            end_date=unique_trade_date[iter_num])

    trading_environment = DummyVecEnv([lambda: StockEnvTradeTehran(socket=socket,
                                                                   df=trade_data,
                                                                   initial=initial,
                                                                   previous_state=last_state,
                                                                   model_name=name,
                                                                   iteration=iter_num)])
    trading_observation = trading_environment.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(trading_observation)
        trading_observation, rewards, dones, info = trading_environment.step(action)

        if i == (len(trade_data.index.unique()) - 2):
            last_state = trading_environment.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('trader/drl_stock_trader/results_tehran/last_state_{}_{}.csv'.format(name, i), index=False)

    return last_state
