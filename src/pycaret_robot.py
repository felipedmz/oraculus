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

class PycaretRobot:
    train_filename = 'data/pycaret_best.pickle'
    temp_feat_filename = 'data/temp.features.csv'
    temp_pred_filename = 'data/temp.predictions.csv'
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
    
    def calculate_hurst(self, column):
        column = column.values  # Converte a coluna para um array numpy
        lags = range(2, len(column) // 2)  # Define os tamanhos de janela
        tau = [np.std(np.subtract(column[lag:], column[:-lag])) for lag in lags]
        h = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return h 
    
    def attrib_result_class(self, value):
        category = None
        if (value > 10):
            category = 'positive_strong'
        elif(value > 5 and value <= 10):
            category = 'positive'
        elif(value > 0 and value <= 5):
            category = 'positive_soft'
        elif(value == 0):
            category = 'no_variation'
        elif(value < -10):
            category = 'negative_strong'
        elif(value < -5 and value >= -10):
            category = 'negative'
        elif(value < 0 and value >= -5):
            category = 'negative_soft'
        else:
            category = 'no_category'
        return category
        
    def feature_eng(self, df):
        print(f'\n>>> Feature Eng Entrada={df.columns}')
        rowsCount = len(df)
        #
        last_update = df['datetime'].max()
        print(f'... Trabalhando a partir dos dados de {last_update}')
        """
        calculando a variacao entre as linhas
        """
        df['value_variation'] = df['close'].diff().fillna(0)
        """
        target column
        - baseado no valor anterior definimos uma classe de previsao
        - essa classe ditara o comportamento do robo de trade
        """
        value_class = np.empty(rowsCount, dtype='object')
        for r in range(rowsCount):
            if (r >= 1):
                previous_variation = df.loc[r-1, ['value_variation']].copy()
                value_class[r] = self.attrib_result_class(previous_variation[0])
        df['value_class'] = value_class
        """
        date features:
        - separamos dia mes e ano, dia da semana
        - ajustados os tipos de dados
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.strftime('%Y').astype('int')
        df['month'] = df['datetime'].dt.strftime('%m').astype('int')
        df['day'] = df['datetime'].dt.strftime("%d").astype('int')
        df['hour'] = df['datetime'].dt.strftime('%H')
        df['minute'] = df['datetime'].dt.strftime('%M')
        df['week_day'] = df['datetime'].dt.strftime('%A')
        df.drop(columns=['datetime'], inplace=True)
        """
        coeficients features
        - razao entre o volume traded e disponivel
        - amplitude de variacao high-low
        - "candle" abstraido como a diferenca entre open-close
        """
        df['traded_volume'] = df['number_of_trades'] / df['volume']
        df['amplitude'] = df['high'] - df['low']
        df['candle'] = df['close'] - df['open']
        #
        """
        features usando o expoente de Hurst
        @see: https://en.wikipedia.org/wiki/Hurst_exponent
        - o hurst carrega a memoria da serie temporal
        - tambem usado como medida de aletoriedade
        - criado indicador hurst para variacao, volume, amplitude e candle
        - hurst de cada linha calculado usando as 10 (dez) linhas anteriores
        """
        df.reset_index(drop=True, inplace=True)
        #
        h_value_variation = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r >= 10):
                lasts = df.loc[r-10:r, ['value_variation']].copy()
                h = self.calculate_hurst(lasts)
                h_value_variation[r] = h
        df['h_value_variation'] = h_value_variation
        #
        h_traded_volume = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r >= 10):
                lasts = df.loc[r-10:r, ['traded_volume']].copy()
                h = self.calculate_hurst(lasts)
                h_traded_volume[r] = h
        df['h_traded_volume'] = h_traded_volume
        #
        h_amplitude = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r >= 10):
                lasts = df.loc[r-10:r, ['amplitude']].copy()
                h = self.calculate_hurst(lasts)
                h_amplitude[r] = h
        df['h_amplitude'] = h_amplitude
        #
        h_candle = np.zeros(rowsCount)
        for r in range(rowsCount):
            if (r >= 10):
                lasts = df.loc[r-10:r, ['candle']].copy()
                h = self.calculate_hurst(lasts)
                h_candle[r] = h
        df['h_candle'] = h_candle
        """
        [end] Hurst Statistics
        """
        # filter
        df = df[df['h_value_variation'] != 0]
        print(f'>>> Feature Eng Saída={df.columns}\n')
        df.to_csv(self.temp_feat_filename, index=False)
        #
        return df
    
    def train(self):
        print(f'\n>>> Etapa Treinamento')
        # 
        print(f'... carregando treinamento')
        df = pd.read_csv('data/quotation.csv')
        df = self.feature_eng(df)
        print(f'... debug de features salvo em = {self.temp_feat_filename}')
        # init setup
        print('... setup')
        exp = setup(
            data=df,
            target='value_class',
            train_size = 0.8,
            fold_shuffle=True,
            session_id=123
        )
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
        #
        while self.check_execution():
            last_ocurrencies = self.api.cripto_quotation()
            to_predict = self.feature_eng(last_ocurrencies)
            print(f'... Prevendo as ultimas {len(to_predict)} ocorrencias...')
            predictions = predict_model(model, data=to_predict)
            predictions.to_csv(self.temp_pred_filename, index=False)            
            self.await_next_iteraction()
