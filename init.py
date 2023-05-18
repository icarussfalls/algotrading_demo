from binance.client import Client
import json
import time
from common import *

with open("configure.json", "r") as json_file:
    json_data = json.load(json_file)


class login():
        global client
        api_key = json_data.get('api_key')
        api_secret = json_data.get('api_secret')

        while True:
            try:
                client = Client(api_key, api_secret)
                print('Client connected...')
                break
            except:
                print('Client is reconnecting in 5 seconds')
                time.sleep(5)
                continue

