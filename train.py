import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from src.simple_robot import feature_eng
# Baixando os dados de DOGE COIN
df = pd.read_parquet('data/BTC-USDT.parquet')

#### Calculando o target (y)

# Calculando qual a média de close dos próximos 10min
df['forward_average'] = df[::-1]['close'].rolling(10).mean()[::-1].shift(-1)

# Target será a diferença percentual do 'forward_average' com o 'close' atual
df['target'] = 100*(df['forward_average'] - df['close']) / df['close']

df.head(3)
# Outra possibilidade: target como a diferença entre o proximo minuto e o atual: df['diff']= -df['close'].diff(-1)

#### Calculando as features (x)
# Toda a parte de criação de features está no arquivo simple_robot.py. Aqui apenas chamamos a função. Isso é útil, pois conseguimos usar a mesma função no momento de colocar o robô em produção
df = feature_eng(df)
df.head(5)

#### Separando em treino/ teste
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

# Modelo linear simples
model = sm.OLS(y_train,X_train).fit()
print(model.summary())
print('\n\n')

#### Resultado do Modelo Linear
y_hat = model.predict(X_test)
MSE = ((y_hat - y_test)**2).mean()
print(f'MSE={MSE}')

MAE = ((y_hat - y_test).abs()).mean()
print(f'MAE={MAE}')

#### Referência
"""
É sempre recomendado ter valores de referência, para saber se seu modelo é ou não melhor do que outras alternativas "naive"
Abaixo, um exemplo de resultado Naive, assumindo todos 0
"""
MSE_assuming_all_zero = (y_test**2).mean()
print(f'MSE_assuming_all_zero={MSE_assuming_all_zero}')

MAE_assuming_all_zero = (y_test.abs()).mean()
print(f'MAE_assuming_all_zero={MAE_assuming_all_zero}')

# Salvando o modelo em um arquivo pickle para ser utilizado nas etapas seguintes
filename = 'data/model_dummy.pickle'
pickle.dump(model, open(filename, 'wb'))

print(f'\nExport realizado para: {filename}')
