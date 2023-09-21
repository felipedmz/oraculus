import sys

from src.api import Client
from datetime import datetime

import statsmodels.api as sm
import pandas as pd
import numpy as np

import math
import pickle
import time

from pycaret.classification import *
from hurst import compute_Hc

class PycaretRobot:
    train_filename = 'data/pycaret_best.pickle'
    temp_feat_filename = 'data/temp.features.csv'
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
        
    def calculate_hurst(self, subset: pd.DataFrame):
        data = subset.values
        H, c, data = compute_Hc(data, kind='random_walk')
        return H    
        
    def feature_eng(self, df):
        print(f'\n>>> Feature Eng Entrada={df.columns}')
        #
        last_update = df['datetime'].max()
        print(f'... Trabalhando a partir dos dados de {last_update}')
        # target column
        df['value_variation'] = df['close'].diff().fillna(0)
        # date features
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.strftime('%Y')
        df['month'] = df['datetime'].dt.strftime('%m')
        df['day'] = df['datetime'].dt.strftime("%d")
        df['hour'] = df['datetime'].dt.strftime('%H')
        df['minute'] = df['datetime'].dt.strftime('%M')
        df['week_day'] = df['datetime'].dt.strftime('%A')
        df.drop(columns=['datetime'], inplace=True)
        # coeficients
        df['traded_volume'] = df['number_of_trades'] / df['volume']
        df['amplitude'] = df['high'] / df['low']
        df['candle'] = df['close'] / df['open']
        #
        """
        new columns with Hurst Statistics
        @see: https://en.wikipedia.org/wiki/Hurst_exponent
        """
        rowsCount = len(df)
        #
        h_value_variation = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r > 100):
                lasts = df.loc[r-100:r, ['value_variation']].copy()
                h = self.calculate_hurst(lasts)
                h_value_variation[r] = h
        df['h_value_variation'] = h_value_variation
        #
        h_traded_volume = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r > 100):
                lasts = df.loc[r-100:r, ['traded_volume']].copy()
                h = self.calculate_hurst(lasts)
                h_traded_volume[r] = h
        df['h_traded_volume'] = h_value_variation
        #
        h_amplitude = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r > 100):
                lasts = df.loc[r-100:r, ['amplitude']].copy()
                h = self.calculate_hurst(lasts)
                h_amplitude[r] = h
        df['h_amplitude'] = h_value_variation
        #
        h_candle = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r > 100):
                lasts = df.loc[r-100:r, ['candle']].copy()
                h = self.calculate_hurst(lasts)
                h_amplitude[r] = h
        df['h_candle'] = h_candle
        """
        [end] Hurst Statistics
        """
        # filter
        df = df[df['h_value_variation'] > 0]
        print(f'>>> Feature Eng Saída={df.columns}\n')
        return df
    
    def train(self):
        print(f'\n>>> Etapa Treinamento')
        # 
        print(f'... carregando treinamento')
        df = pd.read_csv('data/quotation.csv')
        df = self.feature_eng(df)
        df.to_csv(self.temp_feat_filename, index=False)
        print(f'... debug de features salvo em = {self.temp_feat_filename}')
        sys.exit(1)
        # init setup
        print('... setup')
        exp = setup(data=df, target=df['value_variation'])
        # compare models
        print('... compare models')
        best_model = compare_models()
        print('... best model')
        print(best_model)
        final_model = create_model(best_model)
        # metrics
        print('\n>>> Avaliação de modelo')
        evaluate_model(final_model)
        # para salvar o modelo
        save_model(final_model, self.train_filename)
        print(f'\n>>> Export realizado para: {self.train_filename}')
        
    def setTime(self, time):
        self.time = time
        
    def execute(self, time: int):
        self.setTime(time)
        print(f'\n>>> Realizando trades')
        # carregando o aprendizado do modelo
        model = load_model(self.train_filename)
        last_ocurrencies = self.api.cripto_quotation()
        while self.check_execution():
            predictions = predict_model(model, data=last_ocurrencies)
            print(predictions)
            self.await_next_iteraction()
