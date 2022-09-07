# DRL_Stock_Trader
This is my BSc. Final Project which is about using Deep Reinforcement Learning for stock trading.

By Kasra Mojallal


## About the Project

This Project is about using Deep Reinforcement Learning for stock trading. The main dataset of the project is dow-jones data from 2009 to 2020.

Since we are using RL, gym library from OpenAI was used for creating the environments for stock trading. Also, the stable baselines 3 was used for creating the algorithms.


## Creating the virtual environment
It is highly recommended to create a virtual environment for your project. To do so, you can create it using the requirements.txt file:

$ pip install requirements.txt

Also, please use pip instead of conda for creating the environment.


## Using the web-app
This is a Django Project, to run it you have to do:

$ python manage.py runserver

From there, you can register your name and choose from Dow or Tehran stock markets. When you open the page for one of the traders, you can select the hyper-parameters for starting the project.

The initial amount selects the amount of money that the algorithm has before the trading process begins
The train start date selects the starting date for training process
The start and end dates for trading select the dates that the real trading are happening in between
The algorithm robustness shows how many steps would the algorithm take to finish each epoch of the training process


The logs of the systems will be printed out on the right hand and the online results are on the bottom.


## Using the stock-trader code
It is also possible to skip using the web-app code and run the code from the terminal. For doing that, you have to open the terminal in "./stock_trader_website/trader/drl_stock_trader" and then run this code:

$ python main.py

The code will not log the results in the terminal but if you add print, next to the SocketIO messages, you will get the results that you want.


## Explaining the code

./

The structure of the code is a Django project so it has apps in it.
The base app is stock_trader_website
The trader app is where the drl_stock_trader code is


./trader/drl_stock_trader

Folders -->

The dataset folder has the datasets for dow-jones and tehran stocks, both the raw data and pre-processed ones
The results and results_tehran data will contain the results and logs of the code after you run it
The RL_envs and RL_envs_tehran folders contain the environments for training, validation and testing, both for dow-jones and tehran data.
The trained_models and trained_models_tehran will contain the saved algorithms after they were trained on trained data.


Files -->

main.py will choose the main dataset of the system (dow-jones or tehran stocks), pre-processes the datasets if they haven't been pre-processed and will create the dates for train and trading periods.
data_retriever.py will get the raw data.
preprocess.py will pre-process the data.
models.py will create the environments, the algorithms and make the algorithms train on the environments.
algorithms.py contains the RL algorithms and the process of trading

