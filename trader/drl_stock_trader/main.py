import os
import pandas as pd
import warnings

from trader.drl_stock_trader import data_retriever
from trader.drl_stock_trader import preprocess
from trader.drl_stock_trader import models

warnings.filterwarnings('ignore')

# path for the pre-processed data
preprocessed_dow_path = "trader/drl_stock_trader/datasets/done_data_dow.csv"
preprocessed_tehran_path = "trader/drl_stock_trader/datasets/done_data_tehran.csv"
merged_stocks_tehran_path = "trader/drl_stock_trader/datasets/tehran_stocks.csv"


def process_input_data(market, initial_amount, robustness, train_start, period_trade):
    market = market
    initial_amount = initial_amount
    robustness = int(robustness)
    train_start = int(train_start)
    trade_start, trade_end = period_trade.split('-')
    trade_start = int(trade_start)
    trade_end = int(trade_end)

    return market, initial_amount, robustness, train_start, trade_start, trade_end


def run_model_offline(socket, market, initial_amount, robustness, train_start, period_trade):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    market, initial_amount, robustness, train_start, trade_start, trade_end = process_input_data(market,
                                                                                                 initial_amount,
                                                                                                 robustness,
                                                                                                 train_start,
                                                                                                 period_trade)

    with open('trader/drl_stock_trader/config/initial_balance.txt', 'w') as f:
        f.write(initial_amount)

    if market == 'dow':

        # if the data are already pre-process then just go for it
        if os.path.exists(preprocessed_dow_path):
            data = pd.read_csv(preprocessed_dow_path, index_col=0)
        # if the data needs to be pre-processed
        else:
            data = preprocess.preprocess_data(market)
            data = preprocess.add_turbulence(data, market)
            data.to_csv(preprocessed_dow_path)

        # get unique dates for trading from 2015/10/01 to 2020/07/07
        # unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        unique_trade_date = data[(data.datadate > trade_start) & (data.datadate <= trade_end)].datadate.unique()

        # ensemble strategy
        models.run_ensemble_strategy(socket=socket,
                                     df=data,
                                     unique_trade_date=unique_trade_date,
                                     train_start=train_start,
                                     robustness=robustness)



    elif market == 'tehran':

        if not os.path.exists(merged_stocks_tehran_path):
            data_retriever.merge_all_stocks('./tickers')

        # if the data are already pre-process then just go for it
        if os.path.exists(preprocessed_tehran_path):
            data = pd.read_csv(preprocessed_tehran_path, index_col=0)
        # if the data needs to be pre-processed
        else:
            data = preprocess.preprocess_data(market)
            # data = preprocess.manage_missing_data_for_tehran_stocks(data)
            data.to_csv(preprocessed_tehran_path)

        data = data.loc[data['tic'] == 'آ س پ-ت']
        start_date = train_start
        unique_trade_date = data[data.datadate > trade_start].datadate.unique()

        # ensemble strategy
        models.run_ensemble_tehran(socket=socket,
                                   df=data,
                                   start_date=start_date,
                                   unique_trade_date=unique_trade_date,
                                   robustness=robustness)
