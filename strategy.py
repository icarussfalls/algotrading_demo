from regime import *
from predicted import *
from trades import *
from init import *
from common import *

with open("configure.json", "r") as json_file:
    json_data = json.load(json_file)

def get_tick_size(symbol: str) -> float:
    info = client.futures_exchange_info()

    for symbol_info in info['symbols']:
        if symbol_info['symbol'] == symbol:
            for symbol_filter in symbol_info['filters']:
                if symbol_filter['filterType'] == 'PRICE_FILTER':
                    return float(symbol_filter['tickSize'])

def get_rounded_price(symbol: str, price:float) -> float:
    return round_step_size(price,get_tick_size(symbol))



def gethourlydata_(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback, klines_type=HistoricalKlinesType.FUTURES ))
    frame = frame.iloc[:, :6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit = 'ms')
    frame = frame.astype(float)
    frame = frame.iloc[:,:]
    return frame


def get_precision(symbol):
    info = client.futures_exchange_info()
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']




class Strategy():
    def __init__(self, pair):
        self.pair = pair
        self.regime_detect = Regime_Detection(pair)  #regime()
        self.preds = data(pair)  #calculate_rolling_average()
        self.trades = Trades(pair)
        self.cooldown_period = 15 * 60 #15mins cooldown
        self.last_trade_time = 0

    def position_size(self, account_balance, risk_tolerance, stddev, leverage):
        risk_amount = account_balance * (risk_tolerance / 100)
        max_position_size = account_balance * leverage
        position_size = risk_amount / (stddev * math.sqrt(10)) #10 periods
        position_size = min(position_size, max_position_size)
        return position_size

    
    
    def current_balance(self):
        now_balance = client.futures_account_balance()
        dt = pd.DataFrame(now_balance)
        balance = float(dt.loc[dt['asset']=='USDT']['balance'])
        account_bal = float(balance)
        return account_bal

    def position(self):
        df = gethourlydata_(self.pair, '15m', '3days')
        df.ta.stdev(append=True)
        account_balance = self.current_balance()
        risk_tolerance = 1
        stddev = df.STDEV_30[-1]
        leverage = json_data['leverage']
        position_sizee = self.position_size(account_balance, risk_tolerance, stddev, leverage)
        return position_sizee

    def calculate_atr(self, lookback=14):
        df = gethourlydata_(self.pair, '15m', '24h')
        tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()))
        tr = np.maximum(tr, np.abs(df['Low'] - df['Close'].shift()))
        atr = tr.ewm(span=lookback, adjust=False).mean()
        time.sleep(0.2)
        return atr.iloc[-1]

    
    def limit_price(self):
        order_book = client.futures_order_book(symbol=self.pair)
        ask = order_book["asks"][0][0]
        bid = order_book["bids"][0][0]
        return float(ask), float(bid)  
    
    def buy(self):
        #code for buying

    def sell(self):
        #code for selling

    def check_cooldown_period(self):
        if self.last_trade_time == 0:
            return 0
        else:
            current_time = time.time()
            time_since_last_trade = current_time - self.last_trade_time
            if time_since_last_trade < self.cooldown_period:
                return 1
            else:
                self.last_trade_time = 0
                return 0

    
    def decision(self):
        open_orders = client.futures_get_open_orders(symbol=self.pair)
        if len(open_orders) != 0:
            print('cancelling the existing orders')
            client.futures_cancel_all_open_orders(symbol=self.pair)
            time.sleep(2)
        #check whether the rolling average is None
        df = self.preds.rolling_average
        if df == None:
            print('at least 10 datas required to calculate the rolling average')
            time.sleep(30)
            return
        pos_size = client.futures_position_information()
        d_size = pd.DataFrame(pos_size)
        pos = (d_size.loc[d_size['symbol'] == str(self.pair)])
        qty = float(pos['positionAmt'])
        pos_amount = round(qty, get_precision(self.pair))
        regime = self.regime_detect.regime
        print('regime is ' + str(regime))
        print('predicted return is ' + str(df))
        time.sleep(3)

        if pos_amount != 0:
            print("existing positions detected")

            if pos_amount > 0:
                print('Long position detected')
                current_price = self.limit_price()[1]
                entry_price = self.trades.get_last_entry_price()
                regime = self.regime_detect.regime

                # calculate trailing stop loss price
                atr = self.calculate_atr()
                # calculate trailing stop loss price
                if regime == 1:
                    atr_multiplier = 3.0
                else:
                    atr_multiplier = 2.0
                atr = self.calculate_atr()
                stop_loss_price = round(entry_price - atr_multiplier * atr, get_precision(self.pair))
                pred = self.preds.rolling_average
                while current_price >= stop_loss_price or regime != -1:
                    atr = self.calculate_atr()
                    stop_loss_price = max(stop_loss_price, round(current_price - atr_multiplier * atr, get_precision(self.pair)))
                    current_price = self.limit_price()[0]
                    regime = self.regime_detect.regime
                    pred = self.preds.rolling_average
                    trailing_sl = self.preds.calculate_trailing_stop()
                    print('Current price', current_price)
                    print('Stop_loss', stop_loss_price)
                    if regime == 1:
                        atr_multiplier = 3.0
                    else:
                        atr_multiplier = 2.0
                    print('regime is ' + str(regime))
                    print(f"Current Price: {current_price}, Stop Loss Price: {stop_loss_price}")
                    #print('trailing_sl', trailing_sl)
                    time.sleep(0.5)
                    
                    if regime == -1:
                        print('Trade exited on regime change')
                        break
                    elif pred == -1:
                        print('Trade exited on pred change')
                        break
                    elif current_price <= stop_loss_price:
                        print('Trade exited on ATR')
                        break

                print('Trade exited.')
                client.futures_create_order(symbol=self.pair, side='SELL', type='MARKET', quantity=pos_amount, reduceOnly=True)
                self.last_trade_time = time.time()

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                exit_price = self.limit_price()[1]  # Get the exit price

                self.trades.update_last_trade_exit_price(exit_price)  # Update the exit price of the last trade

                data = {}
                data['time'] = str(time.ctime())
                data['long_exit_price'] = self.limit_price()[1]
                data['qty'] = str(pos_amount)
                with open('trades.json', 'a') as file:
                    json.dump(data, file)
                    file.write("\n")

                return

            elif pos_amount < 0:
                pos_amount = abs(pos_amount)
                print('Short position detected')
                current_price = self.limit_price()[0]
                entry_price = self.trades.get_last_entry_price()

                # calculate trailing stop loss price

                if regime == -1:
                    atr_multiplier = 3.0
                else:
                    atr_multiplier = 2.0
                atr = self.calculate_atr()
                stop_loss_price = round(entry_price + atr_multiplier * atr, get_precision(self.pair))
                pred = self.preds.rolling_average
                while current_price <= stop_loss_price or regime != 1:
                    atr = self.calculate_atr()
                    stop_loss_price = min(stop_loss_price, round(current_price + atr_multiplier * atr, get_precision(self.pair)))
                    current_price = self.limit_price()[0]
                    regime = self.regime_detect.regime
                    pred = self.preds.rolling_average
                    trailing_sl = self.preds.calculate_trailing_stop()
                    print('Current price', current_price)
                    print('Stop_loss', stop_loss_price)
                    if regime == -1:
                        atr_multiplier = 3.0
                    else:
                        atr_multiplier = 2.0
                    print('regime is ' + str(regime))
                    print(f"Current Price: {current_price}, Stop Loss Price: {stop_loss_price}")
                    #print('trailing_sl', trailing_sl)
                    time.sleep(0.5)

                    if regime == 1:
                        print('Trade exited on regime change')
                        break
                    elif pred == 1:
                        print('Trade exited on pred change')
                        break
                    elif current_price >= stop_loss_price:
                        print('Trade exited on ATR')
                        break

                print('Trade exited.')
                client.futures_create_order(symbol=self.pair, side='BUY', type='MARKET', quantity=pos_amount, reduceOnly=True)
                self.last_trade_time = time.time()

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                exit_price = self.limit_price()[0]  # Get the exit price
                self.trades.update_last_trade_exit_price(exit_price)  # Update the exit price of the last trade
                
                data = {}
                data['time'] = str(time.ctime())
                data['short_exit_price'] = self.limit_price()[0]
                data['qty'] = str(pos_amount)
                with open('trades.json', 'a') as file:
                    json.dump(data, file)
                    file.write("\n")
                return

        else:
            if self.check_cooldown_period() == 0:
                if self.preds.rolling_average == 1 and self.regime_detect.regime == 1:
                    print(df)
                    self.buy()
                    return

                elif self.preds.rolling_average == -1 and self.regime_detect.regime == -1:
                    print(df)
                    self.sell()
                    return 
            
    def test(self):
        print(self.preds.calculate_rolling_average())
        time.sleep(20)



#if __name__ == '__main__':
    #y = Strategy('BTCUSDT')
    #print(y.position())




    
