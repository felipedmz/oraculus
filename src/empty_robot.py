# @usage: from src.empty_robot import EmptyRobot

from src.api import Client
from datetime import datetime

import statsmodels.api as sm
import pandas as pd
import numpy as np

import math
import pickle
import time

class EmptyRobot:
    train_filename = 'data/ALTERAR.pickle'

    # common
    api = None
    time = 0
    current_execution = 0
    #
    def __init__(self, api: Client):
        print(f'\n>>> Este é um robo vazio para você usar de modelo')
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
        
        print(f'>>> Feature Eng Saída={df.columns}\n')
        return df
    
    def train(self):
        print(f'\n>>> Etapa Treinamento')
        
        # seu codigo
        
        # para salvar o modelo       
        #pickle.dump(model, open(self.train_filename, 'wb'))
        
        print(f'\n>>> Export realizado para: {self.train_filename}')
        
    def execute(self, time: int):
        self.time = time
        print(f'\n>>> Realizando trades')
        
        # carregando o aprendizado do modelo
        # model = pickle.load(open(self.train_filename, 'rb'))
        ticker = 'BTC'        
        while self.check_execution():
            # seu codigo
            #
            self.await_next_iteraction()
