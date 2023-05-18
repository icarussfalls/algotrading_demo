from init import *
from regime import *
from predicted import *
from strategy import *


global pairs
pairs = ['pair_name']
active = True

def trade(i):
    global active
    login()
    session = Strategy(i)
    while active:
        try:
            session.decision()
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            login()
            continue


if __name__ == '__main__':
    trade('')


