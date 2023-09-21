import sys
import pandas as pd
from datetime import datetime, timedelta

from src.api import Client

filename = 'data/quotation.csv'
api = Client('prd')

print(f'[{datetime.now()}] > Atualização de Cotações')
try:
    existing_df = pd.read_csv(filename)
except:
    print('... Arquivo não existente para atualização, armazenando as últimas cotações.')
    last_quotations = api.cripto_quotation()
    last_quotations.to_csv(filename, index=False)
    print(f'[{datetime.now()}] > End')
    sys.exit(1)

print(f'... Cotações armazenadas = {len(existing_df)}')
existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
last_update = existing_df['datetime'].max()

new_quotations = api.cripto_quotation()
print(f'... Cotações obtidas online = {len(new_quotations)}')
new_filtered = new_quotations[new_quotations['datetime'] > last_update]
if len(new_filtered) == 0:
    print('... Sem novos preços para atualizar')
    sys.exit(1)
    
print(f'... Cotações armazenadas = {len(existing_df)}')
print(f'... Última atualização: {last_update} | {len(new_filtered)} novas cotações para armazenar')
    
merged = pd.concat([existing_df, new_filtered], ignore_index=True)
merged.to_csv(filename, index=False)
print(f'... Novo total de cotações armazenadas = {len(merged)}')
#
print(f'[{datetime.now()}] > End')
