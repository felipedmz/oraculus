from src.api import Client
from datetime import datetime

import statsmodels.api as sm
import pandas as pd
import numpy as np

import math
import pickle
import time

class SimpleRobot:
    train_filename = 'data/simple_robot.pickle'
    api = None
    
    # common
    time = 0
    current_execution = 0
    #
    def __init__(self, api: Client):
        print(f'\n>>> Inicializando')
        self.api = api
        print(f'...... time = {self.time}')
        print(f'...... current_execution = {self.current_execution}')
        
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
        
        # Duas variáveis lagged percentual:
        df['lag_1']= 100*df['close'].diff(1)/df['close']
        df['lag_2']= 100*df['close'].diff(2)/df['close']

        # Moving averages e razões entre elas
        df['ma_10']= df['close'].rolling(10).mean()
        df['ma_30']= df['close'].rolling(30).mean()
        df['ratio_ma'] = df['ma_10']/df['ma_30']

        # Um contador de minutos desde o início da série
        df['time']=(df['datetime'].astype(np.int64)/6e10).astype(int)
        cols2drop = set(['symbol', 'datetime', 'close_time', 'open', 'high', 'low', 'forward_average']).intersection(set(df.columns))
        df = df.drop(columns=cols2drop)
        df = df.dropna()
        df['time'] = df['time']-26038829
        
        print(f'>>> Feature Eng Saída={df.columns}\n')
        return df
    
    def train(self):
        print(f'\n>>> Etapa Treinamento')
        
        ### 1) Baixando os dados de BTC
        df = pd.read_parquet('data/BTC-USDT.parquet')
        #df = pd.read_parquet('https://drive.google.com/u/0/uc?id=17c2r9qbnsxPVxaYukrp6vhTY-CQy8WZa&export=download')

        ### 2) Calculando o target (y)
        # Calculando qual a média de close dos próximos 10min
        df['forward_average'] = df[::-1]['close'].rolling(10).mean()[::-1].shift(-1)
        # Target será a diferença percentual do 'forward_average' com o 'close' atual
        df['target'] = 100*(df['forward_average'] - df['close']) / df['close']
        df.head(3)
        
        """ 
        Outra possibilidade: target como a diferença entre o proximo minuto e o atual: 
        df['diff']= -df['close'].diff(-1)
        """

        ### 3) Calculando as features (x)
        """ 
        Toda a parte de criação de features está no arquivo simple_robot.py. 
        Aqui apenas chamamos a função. Isso é útil, pois conseguimos usar a mesma função no momento 
        de colocar o robô em produção
        """
        
        df = self.feature_eng(df)
        df.head(5)

        ### 4) Separando em treino/ teste
        """ 
        Separando usando data. Isso é importante, pois precisamos entender se os modelos criados 
        em um tempo passado continua sendo útil em um tempo futuro.
        """
        test_treshold = '2021-06-01 00:00:00'

        train = df[df.index <= test_treshold]
        test = df[df.index > test_treshold]

        X_train = train.drop(columns=['target'])
        y_train = train['target']

        X_test = test.drop(columns=['target'])
        y_test = test['target']

        ### 5)  Modelo linear simples
        model = sm.OLS(y_train,X_train).fit()
        print(model.summary())
        print('\n\n')

        ### 6) Resultado do Modelo Linear
        y_hat = model.predict(X_test)
        MSE = ((y_hat - y_test)**2).mean()
        print(f'MSE={MSE}')

        MAE = ((y_hat - y_test).abs()).mean()
        print(f'MAE={MAE}')

        ### 7) Referência
        """
        É sempre recomendado ter valores de referência, para saber se seu modelo é ou não melhor do que outras alternativas "naive"
        Abaixo, um exemplo de resultado Naive, assumindo todos 0
        """
        MSE_assuming_all_zero = (y_test**2).mean()
        print(f'MSE_assuming_all_zero={MSE_assuming_all_zero}')

        MAE_assuming_all_zero = (y_test.abs()).mean()
        print(f'MAE_assuming_all_zero={MAE_assuming_all_zero}')

        ### 8) Salvando o modelo em um arquivo pickle para ser utilizado nas etapas seguintes
        pickle.dump(model, open(self.train_filename, 'wb'))
        print(f'\n>>> Export realizado para: {self.train_filename}')
        
    def compute_quantity(self, coin_value, invest_value, significant_digits):
        a_number = invest_value/coin_value
        rounded_number =  round(a_number, significant_digits - int(math.floor(math.log10(abs(a_number)))) - 1)
        print(f"rounded_number={rounded_number}")
        return float(rounded_number.iloc[0])
        
    def execute(self, time: int):
        self.time = time
        print(f'\n>>> Realizando trades')
        
        model = pickle.load(open(self.train_filename, 'rb'))
        ticker = 'BTC'
        count_iter = 0
        valor_compra_venda = 10
        
        while self.check_execution():
            # Pegando o OHLC dos últimos 500 minutos
            df = self.api.cripto_quotation()

            # Realizando a engenharia de features
            df = self.feature_eng(df)

            # Isolando a linha mais recente
            df_last = df.iloc[[np.argmax(df['time'])]]

            # Calculando tendência, baseada no modelo linear criado
            tendencia = model.predict(df_last).iloc[0]
            
            # A quantidade de cripto que será comprada/ vendida depende do valor_compra_venda e da cotação atual
            qtdade = self.compute_quantity(coin_value = df_last['close'], invest_value = valor_compra_venda, significant_digits = 2)

            # Print do datetime atual
            print(f'\n...... SimpleRobot @ {datetime.now()}')

            if tendencia > 0.02:
                # Modelo detectou uma tendência positiva
                print(f"Tendência positiva de {str(tendencia)}")

                # Verifica quanto dinheiro tem em caixa
                qtdade_money = self.api.how_much_i_have(ticker)

                if qtdade_money>0:
                    # Se tem dinheiro, tenta comprar o equivalente a qtdade ou o máximo que o dinheiro permitir
                    max_qtdade = self.compute_quantity(coin_value = df_last['close'], invest_value = qtdade_money, significant_digits = 2)
                    qtdade = min(qtdade, max_qtdade)

                    # Realizando a compra
                    print(f'Comprando {str(qtdade)} {ticker}')
                    self.api.buy(qtdade)

            elif tendencia < -0.02:
                # Modelo detectou uma tendência negativa
                print(f"Tendência negativa de {str(tendencia)}")

                # Verifica quanto tem da moeda em caixa
                qtdade_coin = self.api.how_much_i_have(ticker)

                if qtdade_coin>0:
                    # Se tenho a moeda, vou vender!
                    qtdade = min(qtdade_coin, qtdade)
                    print(f'Vendendo {str(qtdade)} {ticker}')
                    self.api.sell(qtdade)
            else:
                # Não faz nenhuma ação, espera próximo loop
                print(f"Tendência neutra de {str(tendencia)}. Nenhuma ação realizada")
            #
            self.await_next_iteraction()
