# coding: utf-8
from common import *
from init import *


with open("configure.json", "r") as json_file:
    json_data = json.load(json_file)

fitted_lambda_dict = joblib.load(json_data['dict'])

def get_tick_size(symbol: str) -> float:
    info = client.futures_exchange_info()

    for symbol_info in info['symbols']:
        if symbol_info['symbol'] == symbol:
            for symbol_filter in symbol_info['filters']:
                if symbol_filter['filterType'] == 'PRICE_FILTER':
                    return float(symbol_filter['tickSize'])

def get_rounded_price(symbol: str, price:float) -> float:
    return round_step_size(price,get_tick_size(symbol))



class ML():
    scaler = StandardScaler()
    def __init__(self, stock_data, model_path):
        self.stock_data = stock_data
        self.model_path = tf.keras.models.load_model(model_path)
        
  
    def pred_values(self):
        data = self.stock_data[-60:]
        next_data = []
        next_data.append(data)
        next_data = np.array(next_data)
        next_test = np.reshape(next_data, (next_data.shape[0], next_data.shape[1], 21))
        price = self.model_path.predict(next_test)
        tf.keras.backend.clear_session()
        return price
    

def gethourlydata(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback, klines_type=HistoricalKlinesType.FUTURES))
    column_names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    frame.columns = column_names
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    frame = frame.drop(['Close time', 'Ignore'], axis=1)  # Drop unrequired columns
    return frame

def calculate_moving_average(df, window=30):
    ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in ohlcv_columns:
        df[column] = df[column].rolling(window=window).mean()
    return df

def drop_zeros(df):
    return df[(df != 0).all(axis=1)]


def get_precision(symbol):
    info = client.futures_exchange_info()
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']

def add_technical_indicators(df):
#code to difference the cols and add technical indicators
    return df



def fill_missing_values(df):
    # Forward fill missing values first
    df = df.ffill()

    # Interpolate any remaining missing values using linear interpolation
    df.interpolate(method='linear', inplace=True)

    return df



def normalize(dataset):
#normalize and scale the datas
    return data_final


class data():
    def __init__(self, pair: str, trend_direction: int = 1):
        self.pair = pair
        self.mu_prior = 0
        self.sigma_prior = 1
        self.window_size = 20
        self.trailing_stop_multiplier = 2
        self.db_returns = sqlite3.connect('returns.db')
        self.returns_table()
        self.pred_return = None
        self.rolling_average = None
        self.threshold_factor_std_dev = None
        self.trend_direction = trend_direction
        self.threshold_factor = None
        self.loop = asyncio.new_event_loop()
        self.loop1 = asyncio.new_event_loop()
        asyncio.ensure_future(self.update_return(), loop=self.loop)
        future = threading.Thread(target=self.loop.run_forever, daemon=True).start()
        asyncio.ensure_future(self.wait_and_calculate_rolling_average(), loop=self.loop1)
        t = threading.Thread(target=self.loop1.run_forever, daemon=True).start()

    def returns_table(self) -> None:
        """Create the 'predicted_returns' table in the SQLite3 database."""
        cursor = self.db_returns.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS predicted_returns
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            return_value REAL)''')
        self.db_returns.commit()

    @staticmethod
    def current_balance(client: Client) -> float:
        now_balance = client.futures_account_balance()
        dt = pd.DataFrame(now_balance)
        balance = float(dt.loc[dt['asset'] == 'USDT']['balance'])
        account_bal = float(balance)
        return account_bal

    @staticmethod
    def path_model_min() -> str:
        with open('configure.json') as f:
            json_data = json.load(f)
        path = json_data['model_path']
        return path

    def column_min(self, df: pd.DataFrame) -> np.ndarray:
        path_model = self.path_model_min()
        pred = ML(df.iloc[:, :], path_model).pred_values()
        scaler_path = json_data.get('train_scaler_predicted')
        scaler = load(scaler_path)
        denormalized_pred = #inverse the pred
        return denormalized_pred
    
    def returns(self):
        # Fetch the data
        df = gethourlydata(self.pair, '15m', '6days')
        df = add_technical_indicators(df)
        
        # Normalize the data
        df_norm = normalize(df)
        
        # Make the prediction using the ML model
        pred = self.column_min(df_norm)
        print('pred', pred)
        return float(pred)

    async def return_async(self) -> float:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await self.loop.run_in_executor(pool, self.returns)
            return result

    async def update_return(self) -> None:
        while True:
            self.pred_return = await self.return_async()
            await self.save_to_database(self.pred_return)
            await asyncio.sleep(15 * 60)  # wait for 15 min to run the model again and save to the database

    async def save_to_database(self, new_pred_return: float) -> None:
        # Prepare the SQL query to insert the new_pred_return value into the database
        db_conn = sqlite3.connect('returns.db')
        sql = "INSERT INTO predicted_returns (return_value) VALUES (?)"
        val = (new_pred_return,)

        # Create a new cursor object for the current thread
        with db_conn: 
            cursor = db_conn.cursor()

            try:
                # Execute the query and commit the transaction
                cursor.execute(sql, val)
                db_conn.commit()
            finally:
                # Always close the cursor object to release the database resources
                cursor.close()


    def observed_returns(self) -> Tuple[np.ndarray, np.ndarray]:
#code to get predicted returns from the database and compare them to observed returns
        return y_pred_filled, y_true



    def get_dynamic_threshold_factor(self) -> float:
        y_pred, y_true = self.observed_returns()
        y_pred = y_pred[:-1]
        #calculate threshold factor

        return threshold_factor

    def update_posterior(self, y):
#code to calculate mu_posterior and sigma_posterior

        return mu_posterior, sigma_posterior


    def calculate_threshold(self) -> float:
        threshold = self.mu_prior + self.threshold_factor * self.sigma_prior
        y_pred, y_true = self.observed_returns()
        y_pred = y_pred[:-1]
        errors = y_pred - y_true

        # Adjust threshold based on standard deviation of errors
        threshold += self.threshold_factor_std_dev * errors_std

        return threshold


    def calculate_trend_direction(self, y_pred, y_true, mu_prior, sigma_prior):
#calculate trend using bayesian updating
        return trend_direction


    def calculate_rolling_average(self):
        y_pred, y_obs = self.observed_returns()
        y_pred = y_pred[1:]

        # Update posterior distribution
        mu_posterior, sigma_posterior = self.update_posterior(y_obs)

        # Get dynamic threshold factor
        threshold_factor = self.get_dynamic_threshold_factor()
        self.threshold_factor = threshold_factor

        # Calculate threshold
        threshold = self.calculate_threshold()

        # Determine trend direction
        trend_direction = self.calculate_trend_direction(y_pred, y_obs, mu_posterior, sigma_posterior)
        self.trend_direction = trend_direction

        print('y_pred', y_pred)
        print('threshold_factor', threshold_factor)
        print('mu_posterior', mu_posterior)
        print('last_y_pred', y_pred[-1])
        print('trend_direction', trend_direction)
        print('threshold', threshold)

        # Determine trade logic based on trend direction and threshold
        if trend_direction == 1 and y_pred[-1] > mu_posterior + threshold:
            return 1  # Buy
        elif trend_direction == -1 and y_pred[-1] < mu_posterior - threshold:
            return -1  # Sell
        else:
            return 0  # Hold


        

    def calculate_trailing_stop(self):
        # Get observed and predicted returns
        y_pred, y_obs = self.observed_returns()
        y_pred = y_pred[1:]

        # Calculate rolling standard deviation of predicted returns
        rolling_std = np.std(y_pred)

        # Determine trend direction
        trend_direction = self.trend_direction

        # Calculate trailing stop based on trend direction
        if trend_direction == 1:
            trailing_stop_long = y_pred[-1] - self.trailing_stop_multiplier * rolling_std
            trailing_stop_short = None
        elif trend_direction == -1:
            trailing_stop_long = None
            trailing_stop_short = y_pred[-1] + self.trailing_stop_multiplier * rolling_std
        else:
            trailing_stop_long = None
            trailing_stop_short = None

        # Get current percentage change
        current_pct_change = y_obs[-1]

        # Determine if stop loss is triggered for long and short positions
        if trailing_stop_long is not None and current_pct_change < trailing_stop_long:
            return 1
        elif trailing_stop_short is not None and current_pct_change > trailing_stop_short:
            return -1
        else:
            return 0



    async def wait_and_calculate_rolling_average(self):
            while True:
                # Connect to the SQLite3 database
                conn = sqlite3.connect('returns.db')
                cursor = conn.cursor()

                # Check the count of predicted returns in the database
                cursor.execute("SELECT COUNT(*) FROM predicted_returns")
                count = cursor.fetchone()[0]

                if count >= 10:
                    # Calculate the rolling average and return the result
                    result = self.calculate_rolling_average()
                    self.rolling_average = result
                    await asyncio.sleep(15 * 60)

                else:
                    # Wait for 10 seconds before checking again
                    await asyncio.sleep(10)



    def last_return(self):
        try:
            conn = sqlite3.connect('returns.db')

            # Query the last 50 rows from predicted_returns
            query = "SELECT * FROM predicted_returns ORDER BY id DESC LIMIT 50"

            # Fetch the results and convert to a Pandas DataFrame
            df = pd.read_sql_query(query, conn)

            # Get the last predicted return
            predicted_return = df['return_value'].iloc[-1]

            # Close the database connection
            conn.close()

            return predicted_return
        except Exception as e:
            print("Error retrieving predicted returns:", e)

    def aveg(self):
        while True:
            print(self.calculate_rolling_average())
            time.sleep(5)
