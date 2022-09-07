import time
import json
from tracemalloc import stop
from urllib import robotparser

from trader.drl_stock_trader import algorithms
from trader.drl_stock_trader.preprocess import data_split

from stable_baselines3.common.vec_env import DummyVecEnv

from trader.drl_stock_trader.RL_envs.EnvMultipleStocks_Train import StockEnvTrain
from trader.drl_stock_trader.RL_envs.EnvMultipleStock_Validation import StockEnvValidation

from trader.drl_stock_trader.RL_envs_tehran.env_multiple_train_tehran import StockEnvTrainTehran
from trader.drl_stock_trader.RL_envs_tehran.env_multiple_validation_tehran import StockEnvValidationTehran


def run_ensemble_strategy(socket,
                          df,
                          unique_trade_date,
                          train_start,
                          robustness):
    market = 'dow'

    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    model_use = []

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    if robustness == 1:
        a2c_time_steps = 100
        ppo_time_steps = 100
        ddpg_time_steps = 40
    elif robustness == 2:
        a2c_time_steps = 30000
        ppo_time_steps = 30000
        ddpg_time_steps = 10000
    else:
        a2c_time_steps = 50000
        ppo_time_steps = 50000
        ddpg_time_steps = 30000

    start = time.time()

    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):

        # the first run or not
        if i - rebalance_window - validation_window == 0:
            first_run = True
        else:
            first_run = False

        ############## Environment Setup starts ##############
        ## training environment
        train_data = data_split(df,
                                start_date=train_start,
                                end_date=unique_trade_date[i - rebalance_window - validation_window])
        train_environment = DummyVecEnv([lambda: StockEnvTrain(train_data)])

        ## validation environment
        validation_data = data_split(df,
                                     start_date=unique_trade_date[i - rebalance_window - validation_window],
                                     end_date=unique_trade_date[i - rebalance_window])
        validation_environment = DummyVecEnv([lambda: StockEnvValidation(validation_data, iteration=i)])
        validation_environment_observation = validation_environment.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Model training from: {train_start} to {unique_trade_date[i - rebalance_window - validation_window]}'
        }))
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Model validating from: {unique_trade_date[i - rebalance_window - validation_window]} to {unique_trade_date[i - rebalance_window]}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======A2C Training========'
        }))
        model_a2c = algorithms.train_A2C(socket,
                                         train_environment,
                                         model_name="A2C_{}k_dow_{}".format(a2c_time_steps, i),
                                         timesteps=a2c_time_steps,
                                         market=market)

        algorithms.algorithms_validation(model=model_a2c,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)

        sharpe_a2c = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'A2C Sharpe Ratio: {sharpe_a2c:.2f}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======PPO Training========'
        }))
        model_ppo = algorithms.train_PPO(socket=socket,
                                         env_train=train_environment,
                                         model_name="PPO_{}k_dow_{}".format(ppo_time_steps, i),
                                         timesteps=ppo_time_steps,
                                         market=market)

        algorithms.algorithms_validation(model=model_ppo,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)
        sharpe_ppo = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'PPO Sharpe Ratio: {sharpe_ppo:.2f}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======DDPG Training========'
        }))
        model_ddpg = algorithms.train_DDPG(socket=socket,
                                           env_train=train_environment,
                                           model_name="DDPG_{}k_dow_{}".format(ddpg_time_steps, i),
                                           timesteps=ddpg_time_steps,
                                           market=market)

        algorithms.algorithms_validation(model=model_ddpg,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)
        sharpe_ddpg = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'DDPG Sharpe Ratio: {sharpe_ddpg:.2f}'
        }))

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')

        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Trading from: {unique_trade_date[i - rebalance_window]} to {unique_trade_date[i]}'
        }))
        last_state_ensemble = algorithms.DRL_prediction(socket=socket,
                                                        df=df,
                                                        model=model_ensemble,
                                                        name="ensemble",
                                                        last_state=last_state_ensemble,
                                                        iter_num=i,
                                                        unique_trade_date=unique_trade_date,
                                                        rebalance_window=rebalance_window,
                                                        initial=first_run)
        ############# Trading ends ##############

    end = time.time()
    socket.send(text_data=json.dumps({
        'type': 'terminal',
        'message': f'Ensemble Strategy took: {(end - start) / 60:.2f} minutes'
    }))


def run_ensemble_tehran(socket,
                        df,
                        start_date,
                        unique_trade_date,
                        robustness):
    market = 'tehran'

    last_state_ensemble = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []
    model_use = []

    rebalance_window = 63
    validation_window = 63

    if robustness == 1:
        a2c_time_steps = 100
        ppo_time_steps = 100
        ddpg_time_steps = 40
    elif robustness == 2:
        a2c_time_steps = 30000
        ppo_time_steps = 30000
        ddpg_time_steps = 10000
    else:
        a2c_time_steps = 50000
        ppo_time_steps = 50000
        ddpg_time_steps = 30000

    start = time.time()

    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):

        # the first run or not
        if i - rebalance_window - validation_window == 0:
            first_run = True
        else:
            first_run = False

        ############## Environment Setup starts ##############
        ## training environment
        train_data = data_split(df,
                                start_date=start_date,
                                end_date=unique_trade_date[i - rebalance_window - validation_window])
        train_environment = DummyVecEnv([lambda: StockEnvTrainTehran(train_data)])

        ## validation environment
        validation_data = data_split(df,
                                     start_date=unique_trade_date[i - rebalance_window - validation_window],
                                     end_date=unique_trade_date[i - rebalance_window])
        validation_environment = DummyVecEnv([lambda: StockEnvValidationTehran(validation_data, iteration=i)])
        validation_environment_observation = validation_environment.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Model training from: {start_date} to {unique_trade_date[i - rebalance_window - validation_window]}'
        }))
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Model validating from: {unique_trade_date[i - rebalance_window - validation_window]} to {unique_trade_date[i - rebalance_window]}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======A2C Training========'
        }))
        model_a2c = algorithms.train_A2C(socket=socket,
                                         env_train=train_environment,
                                         model_name="A2C_{}k_dow_{}".format(a2c_time_steps, i),
                                         timesteps=a2c_time_steps,
                                         market=market)

        algorithms.algorithms_validation(model=model_a2c,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)

        sharpe_a2c = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'A2C Sharpe Ratio: {sharpe_a2c:.2f}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======PPO Training========'
        }))
        model_ppo = algorithms.train_PPO(socket=socket,
                                         env_train=train_environment,
                                         model_name="PPO_{}k_dow_{}".format(ppo_time_steps, i),
                                         timesteps=ppo_time_steps,
                                         market=market)

        algorithms.algorithms_validation(model=model_ppo,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)
        sharpe_ppo = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'PPO Sharpe Ratio: {sharpe_ppo:.2f}'
        }))

        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======DDPG Training========'
        }))
        model_ddpg = algorithms.train_DDPG(socket=socket,
                                           env_train=train_environment,
                                           model_name="DDPG_{}k_dow_{}".format(ddpg_time_steps, i),
                                           timesteps=ddpg_time_steps,
                                           market=market)

        algorithms.algorithms_validation(model=model_ddpg,
                                         test_data=validation_data,
                                         test_environment=validation_environment,
                                         test_observation=validation_environment_observation)
        sharpe_ddpg = algorithms.sharpe_ratio_of_algorithms(i, market=market)
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'DDPG Sharpe Ratio: {sharpe_ddpg:.2f}'
        }))

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model Selection based on sharpe ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            model_use.append('A2C')
        else:
            model_ensemble = model_ddpg
            model_use.append('DDPG')

        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        socket.send(text_data=json.dumps({
            'type': 'terminal',
            'message': f'======Trading from: {unique_trade_date[i - rebalance_window]} to {unique_trade_date[i]}'
        }))
        last_state_ensemble = algorithms.DRL_prediction_Tehran(socket=socket,
                                                               df=df,
                                                               model=model_ensemble,
                                                               name="ensemble",
                                                               last_state=last_state_ensemble,
                                                               iter_num=i,
                                                               unique_trade_date=unique_trade_date,
                                                               rebalance_window=rebalance_window,
                                                               initial=first_run)
        ############# Trading ends ##############

    end = time.time()
    socket.send(text_data=json.dumps({
        'type': 'terminal',
        'message': f'Ensemble Strategy took: {(end - start) / 60:.2f} minutes'
    }))
