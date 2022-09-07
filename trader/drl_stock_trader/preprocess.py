import numpy as np
import pandas as pd
from stockstats import StockDataFrame

from trader.drl_stock_trader.config import config


# split data from start_data to end_date
def data_split(df, start_date, end_date):
    
    # split data from start_date to end_date
    data = df[(df.datadate >= start_date) & (df.datadate < end_date)]
    
    # sort values with respect to datadate column and tic column (data are already sorted though)
    data = data.sort_values(['datadate','tic'], ignore_index=True)
    
    # sort the indexes that the same dates have the same index
    data.index = data.datadate.factorize()[0]
    
    return data


def preprocess_data(market):
    
    if market == 'dow':
    
        # read the raw data
        df = pd.read_csv(config.TRAINING_DATA_FILE_DOW)
        
        # normalize data
        df = calculate_price(df)
        
        # add technical indicators
        df = add_technical_indicator(df)
        
        # fill the NULL data
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    elif market == 'tehran':
        
        # read the raw data
        df = pd.read_csv(config.TRAINING_DATA_FILE_TEHRAN)
        
        # normalize data
        df = calculate_price(df)
        
        # add technical indicators
        df = add_technical_indicator(df)
        
        # fill the NULL data
        df.fillna(method='bfill', inplace=True)
        
        return df
        


def add_turbulence(df, market):
    
    if market == 'dow':
        
        turbulence_index = calculate_turbulence(df)
        
        df = df.merge(turbulence_index, on='datadate')
        df = df.sort_values(['datadate','tic']).reset_index(drop=True)
        
        return df



def calculate_price(df):
    
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    
    # if data is 0 then change to 1, else just let it go
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    # the adjusted closed price
    data['adjcp'] = data['prccd'] / data['ajexdi']
    
    # the open price
    data['open'] = data['prcod'] / data['ajexdi']
    
    # the high price
    data['high'] = data['prchd'] / data['ajexdi']
    
    # the low price
    data['low'] = data['prcld'] / data['ajexdi']
    
    # the volume of 
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    
    return data


def add_technical_indicator(df):
    
    # using stock stats wrapper on our data frame
    stock = StockDataFrame.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    
    # iterate for each ticker
    for i in range(len(unique_ticker)):
        
        ## Moving Average Convergence Divergence
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        
        ## Relative Strength Index
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        
        ## Commodity Channel Index
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        
        ## Trend Strength Indicator
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def calculate_turbulence(df):
    
    # the indexes will be dates, columns will be tics and the values are adjusted prices
    df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
    
    # unique dates
    unique_date = df.datadate.unique()
    
    # start after a year
    forward_start = 252
    turbulence_index = [0] * forward_start
    count = 0
    
    # start from 2010/01/04 to the end
    for i in range(forward_start, len(unique_date)):
        
        # the price of every company in specific date (row is the date and the columns are companies)
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        
        # the prices of every company from start(2009) to specific date (rows are dates and columns are companies)
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        
        # prices (rows are companies and the columns are companies)
        cov_temp = hist_price.cov()
        
        # current_temp is current price minus mean of history of prices
        current_temp = (current_price - np.mean(hist_price, axis=0))
        
        ## TODO (row is specific date and rows are companies)
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
            
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index


def manage_missing_data_for_tehran_stocks(df):
    
    # print(pd.DataFrame(df['datadate'].value_counts()).sort_index())
    
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    
    df = df[(df.datadate > 20140721)]
    
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    
    ##################
    
    return df
