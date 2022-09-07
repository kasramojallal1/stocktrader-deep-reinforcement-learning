import json
from channels.generic.websocket import WebsocketConsumer
from trader.drl_stock_trader.main import run_model_offline


class TradeConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def receive(self, text_data=None, bytes_data=None):
        text_data_json = json.loads(text_data)

        market = text_data_json['market']
        initial_amount = text_data_json['initial_amount']
        robustness = int(text_data_json['robustness'].split('_')[1])
        date_train = text_data_json['date_train'].replace('-', '')
        date_trade_1 = text_data_json['date_trade_1'].replace('-', '')
        date_trade_2 = text_data_json['date_trade_2'].replace('-', '')

        run_model_offline(self,
                          market=market,
                          initial_amount=initial_amount,
                          robustness=robustness,
                          # train_start=date_train,
                          train_start='20090101',
                          # period_trade=f'{date_trade_1}-{date_trade_2}'
                          period_trade=f'{str(20160101)}-{str(20190101)}')

