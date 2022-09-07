import datetime
import os
from pathlib import Path

TRAINING_DATA_FILE_DOW = "trader/drl_stock_trader/datasets/dow_30_2009_2020.csv"
TRAINING_DATA_FILE_TEHRAN = "trader/drl_stock_trader/datasets/tehran_stocks.csv"

now = str(datetime.datetime.now()).replace(":", ".")
TRAINED_MODEL_DIR = f"trader/drl_stock_trader/trained_models/{now}"
# os.makedirs(TRAINED_MODEL_DIR)
Path(TRAINED_MODEL_DIR).mkdir(exist_ok=True, parents=True)

TRAINED_MODEL_DIR_TEHRAN = f"trader/drl_stock_trader/trained_models_tehran/{now}"
# os.makedirs(TRAINED_MODEL_DIR_TEHRAN)
Path(TRAINED_MODEL_DIR_TEHRAN).mkdir(exist_ok=True, parents=True)
