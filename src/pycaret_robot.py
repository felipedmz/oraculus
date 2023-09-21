import sys

from src.api import Client
from datetime import datetime

import statsmodels.api as sm
import pandas as pd
import numpy as np

import math
import pickle
import time

from pycaret.datasets import get_data
from pycaret.classification import *

class PycaretRobot:
    train_filename = 'data/pycaret_best.pickle'

    # common
    api = None
    time = 0
    current_execution = 0
    #
    def __init__(self, api: Client):
        print(f'\n>>> PycaretRobot')
        self.api = api
        
    def check_execution(self) -> bool:
        return self.current_execution < self.time
    
    def await_next_iteraction(self): 
        print(self.api.status())
        self.current_execution +=1
        print(f'...... Aguardando 1 min')
        time.sleep(60)
    # [end] common
        
    def feature_eng(self, df):
        print(f'\n>>> Feature Eng Entrada={df.columns}')
        
        # seu codigo
        # dia, hora, dia da semana
        # hurst
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime_original'].dt.date
        df['time'] = df['datetime_original'].dt.strftime('%H:%M:%S')
        df['week_day'] = df['datetime_original'].dt.strftime('%A')
        df.drop(columns=['datetime'], inplace=True)
               
        #df['hurst'] = pd.to_datetime(df['datetime'])
        #
        df['forward_average'] = df[::-1]['close'].rolling(10).mean()[::-1].shift(-1)
        df['target'] = 100*(df['forward_average'] - df['close']) / df['close']
        
        print(f'>>> Feature Eng Saída={df.columns}\n')
        return df
    
    def train(self):
        print(f'\n>>> Etapa Treinamento')
        
        df = pd.read_csv('data/quotation.csv')
        df = self.feature_eng(df)
        df.head(5)
        
        # init setup
        print('... setup')
        clf1 = setup(data = df, target = 'Class variable')
        print(clf1)

        # compare models
        print('... compare_models')
        best = compare_models()
        print(best)
        
        sys.exit(1)
        
        # para salvar o modelo       
        #pickle.dump(model, open(self.train_filename, 'wb'))
        
        print(f'\n>>> Export realizado para: {self.train_filename}')
        
    def execute(self, time: int):
        self.time = time
        print(f'\n>>> Realizando trades')
        
        # carregando o aprendizado do modelo
        #model = pickle.load(open(self.train_filename, 'rb'))
        ticker = 'BTC'        
        while self.check_execution():
            # seu codigo
            #
            self.await_next_iteraction()
