#!/usr/bin/env python
# coding: utf-8
from common import *
from init import *

with open("configure.json", "r") as json_file:
    json_data = json.load(json_file)



my_precision = tf.keras.metrics.Precision()
my_recall = tf.keras.metrics.Recall()
my_auc = tf.keras.metrics.AUC()

@tf.function(experimental_relax_shapes=True,
              input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                               tf.TensorSpec(shape=(None,3), dtype=tf.float32)])
def my_precision_function(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(tf.squeeze(tf.cast(y_true, tf.int32)), tf.int32), depth=3)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    my_precision.update_state(y_true, y_pred)
    return my_precision.result()

@tf.function(experimental_relax_shapes=True,
              input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                               tf.TensorSpec(shape=(None,3), dtype=tf.float32)])
def my_recall_function(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(tf.squeeze(tf.cast(y_true, tf.int32)), tf.int32), depth=3)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    my_recall.update_state(y_true, y_pred)
    return my_recall.result()

@tf.function(experimental_relax_shapes=True,
              input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.float32),
                               tf.TensorSpec(shape=(None,3), dtype=tf.float32)])
def my_auc_function(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(tf.squeeze(tf.cast(y_true, tf.int32)), tf.int32), depth=3)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    my_auc.update_state(y_true, y_pred)
    return my_auc.result()

class MyModel(tf.keras.Model):
    #model used in the training process
    


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
    time.sleep(1)
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


class Regime_Detection():
    def __init__(self, pair):
        self.pair = pair
        self.regime = 0 #initializing with neutral value
        loop = asyncio.new_event_loop()
        asyncio.ensure_future(self.update_regime(), loop=loop)
        future = threading.Thread(target=loop.run_forever, daemon=True).start()


    def preprocess_data(self, df, window=20, lag=4, n_workers=multiprocessing.cpu_count()):
        # Convert input DataFrame to a Dask DataFrame
        df_dask = dd.from_pandas(df, npartitions=n_workers)

        # Compute the OHLCV changes for each partition
        ohlcv_changes = df_dask.map_partitions(compute_ohlcv_changes)

        # Compute the OHLCV statistics for each partition
        ohlcv_stats = ohlcv_changes.map_partitions(compute_ohlcv_stats, window=window)

        # Compute the autocorrelation of OHLCV changes for each partition
        ohlcv_acf = ohlcv_changes.map_partitions(compute_ohlcv_acf, window=window, lag=lag)
        #calculate and append regime to the dataframe

        # Transform the 'regime' column using the loaded label encoder
        path = json_data.get('encoder')
        label_encoder = joblib.load(path)
        df['regime'] = label_encoder.transform(df['regime'])
        df.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
        df.dropna(inplace=True)
        return df
    
    def regime_(self):
        df = gethourlydata_(self.pair, '15m', '6days')
        print(df.columns)
        df_processed = self.preprocess_data(df)
        df_processed.dropna(inplace=True)
        df_processed = df_processed.iloc[:, :]  # include the last row of the data

        # Load the model
        path = json_data.get('regime')
        model = load_model(path, custom_objects={'MyModel': MyModel,
                                                'my_precision_function': my_precision_function,
                                                'my_recall_function': my_recall_function,
                                                'my_auc_function': my_auc_function})

        # Load the scaler
        scaler_path = json_data.get('train_scaler_regime')
        scaler = load(scaler_path)

        # Preprocess and scale the test data
        df_scaled = scaler.transform(df_processed.iloc[:, :-1].values)  # Exclude last column from scaling
        last_col = df_processed.iloc[:, -1].values.reshape(-1, 1)  # Reshape the unscaled last column
        X = np.concatenate((df_scaled, last_col), axis=1)  # Include last column in X without scaling

        # Reshape the test data to match the input shape expected by the model
        # Adjust the reshaping if necessary based on the expected input shape of the model
        next_data = X.reshape((1, X.shape[0], X.shape[1]))

        # Get predicted class
        y_pred = model.predict(next_data)
        print(y_pred)

        # Get the predicted class index (0, 1, or 2) from the one-hot encoded array
        class_idx = np.argmax(y_pred)

        # Map the class index to the desired output format
        if class_idx == 0:
            return -1  # Sell (downtrend)
        elif class_idx == 1:
            return 0  # Neutral
        else:
            return 1  # Uptrend



    async def regime_async(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, self.regime_)
            return result
        
    async def update_regime(self):
        while True:
                self.regime = await self.regime_async()
                await asyncio.sleep(15 * 60) #wait for 15 min to run the model again and update the regime
                

    def reg(self):
        while True:
            print(self.regime_())
            time.sleep(5)

#if __name__ == '__main__':
    #z = Regime_Detection('BTCUSDT')
    #z.reg()
