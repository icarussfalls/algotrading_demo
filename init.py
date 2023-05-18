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
        #api_key = 'd980990f5357ac5fd52f665aac9b1a0d53d8be09c81b5963a9b2edc98d2fe717'
        #api_secret = 'ef77272802a9b51ea5d1af9f579b5e999199c28dd22d48d1080928dd0df2012c'
        #api_key = json_data.get('testnet_api')
        #api_secret = json_data.get('tesnet_secret')
        while True:
            try:
                client = Client(api_key, api_secret)
                print('Client connected...')
                break
            except:
                print('Client is reconnecting in 5 seconds')
                time.sleep(5)
                continue

