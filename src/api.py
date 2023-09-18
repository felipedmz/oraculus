import requests
from datetime import datetime
import pandas as pd

class Client:
    url = 'https://mighty-bastion-45199.herokuapp.com'
    env = None
    token = None
    
    def __init__(self, env):
        print(f'> Inicializando API Client em={env}')
        if env == 'prd':
            self.env = env
            self.token = 'LKETGE585EOIE5UJG'
        else:
            self.env = 'dev'
            self.token = 'token_dummy_001'
        print(f'> Utilizando o token={self.token}')

    # API POST
    def post(self, route, payload = {}):
        url = self.url + route
        payload['token'] = self.token
        #print(f'url={url}')
        #print(f'payload={payload}')
        response = requests.post(url, data = payload)
        return response

    # API GET
    def get(self, route):
        url = self.url + route
        response = requests.get(url)
        return response
                    
    # Transforma a resposta em json em um pandas.DataFrame
    def parse_response(self, r):
        try:
            result = pd.DataFrame.from_dict(r.json())
            # se coluna datetime estiver presente, converte para datetime
            if 'datetime' in result.columns:
                result['datetime'] = pd.to_datetime(result['datetime'], unit='ms')
        except:
            result = r.text
        return result

    # Verificando cripto disponíveis
    def available_cripto(self):    
        response = self.get('/available_cripto')
        return self.parse_response(response)
    
    # Status da carteira
    def status(self):    
        response = self.post('/status')
        return self.parse_response(response)
    
    # Verificando histórico de todas as movimentações
    def history(self):
        response = self.post('/history')
        return self.parse_response(response)
    
    # Cotação dos últimos 500 minutos (OHLC) de uma cripto
    def cripto_quotation(self):
        payload = {'ticker': 'BTC'}
        response = self.post('/cripto_quotation', payload)
        return self.parse_response(response)

    # Compra de cripto
    def buy(self, qty):
        payload = {
            'ticker': 'BTC', 
            'quantity': qty
        }
        response = self.post('/buy', payload)
        return self.parse_response(response)

    # Vendendo cripto
    def sell(self, qty):
        payload = {
            'ticker': 'BTC', 
            'quantity': qty
        }
        response = self.post('/sell', payload)
        return self.parse_response(response)

    # Quanto ainda existe disponível na carteira
    def how_much_i_have(self, ticker):
        status = self.status()
        status_this_coin = status.query(f"ticker == '{ticker}'")
        if status_this_coin.shape[0] > 0:
            return status_this_coin['quantity'].iloc[0]
        else:
            return 0
    