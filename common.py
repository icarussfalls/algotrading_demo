#!/usr/bin/env python
# coding: utf-8
import asyncio
import threading
from functools import lru_cache
import sqlite3
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas_ta as ta
from hurst import compute_Hc
from decimal import Decimal
from statistics import stdev
from binance.helpers import round_step_size
from multiprocessing import Pool
import logging
import websocket, json, time, datetime, sys, re, os
import pandas as pd
pd.options.mode.chained_assignment = None
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
import tensorflow as tf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential, load_model
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from datetime import datetime, timedelta
import tensorflow as tf
from binance.client import Client
import time
import os
import math
import requests
from joblib import load
from scipy import stats
import pickle
from sklearn.model_selection import KFold
import joblib
import concurrent.futures
import json
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import dask.dataframe as dd
import multiprocessing
import dask
from pykalman import KalmanFilter
from keras.regularizers import l2
from keras.initializers import glorot_uniform
import pandas_ta as ta
from typing import Tuple
from scipy.stats import norm
from functools import lru_cache