import os
from pathlib import Path
import pandas as pd
import pytse_client as tse
import yfinance as yf

    
    
def download_tehran_stock_data():
    tse.download(symbols="all", write_to_csv=True, adjust=True)
    

def merge_all_stocks(tehran_datasets_path):
    file_list = (os.listdir(tehran_datasets_path))
    
    tehran_data_frame = pd.DataFrame(columns=['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd'])
    
    
    for dataset_name in file_list:
        
        if '.csv' not in dataset_name:
            continue 
        
        company_name = dataset_name.replace('.csv', '')
        
        dataset_csv = pd.read_csv(f"{tehran_datasets_path}/{dataset_name}")
        
        dataset_csv.drop(['yesterday', 'value', 'count', 'close'], axis=1)
        
        dataset_csv['ajexdi'] = 1
        dataset_csv['tic'] = company_name
        
        dataset_csv['date'] = dataset_csv['date'].replace('-','', regex=True)
        
        
        data = {'datadate' : dataset_csv['date'], 'tic' : dataset_csv['tic'], 'prccd' : dataset_csv['adjClose'],
                'ajexdi' : dataset_csv['ajexdi'], 'prcod' : dataset_csv['open'], 'prchd': dataset_csv['high'],
                'prcld' : dataset_csv['low'], 'cshtrd' : dataset_csv['volume']}
        dataset_csv_new = pd.DataFrame(data=data)
        
        tehran_data_frame = tehran_data_frame.append(dataset_csv_new, ignore_index=True)
        
        tehran_data_frame.to_csv('./datasets/tehran_stocks.csv')
        
