from src.api import Client
import pandas as pd

env = 'dev'

""" Testes de comunicacao com a API """
def test_get():
    api = Client(env)
    status_code = api.get('/available_cripto').status_code
    assert status_code == 200

def test_post():
    api = Client(env)
    status_code = api.post('/status').status_code
    assert status_code == 200

def test_available_cripto():
    api = Client(env)
    response = api.available_cripto()
    assert response == 'BTC'
    
def test_status():
    api = Client(env)
    response = api.status()
    assert type(response) == pd.core.frame.DataFrame
    
def test_history():
    api = Client(env)
    response = api.history()
    assert type(response) == pd.core.frame.DataFrame

def test_cripto_quotation():
    api = Client(env)
    response = api.cripto_quotation()
    assert type(response) == pd.core.frame.DataFrame

def test_buy():
    api = Client(env)
    qty = 0.2
    response = api.buy(qty)
    assert type(response) == str
    assert f'Compra realizada com sucesso: {qty}' in response
    
def test_sell():
    api = Client(env)
    qty = 0.2
    response = api.sell(qty)
    assert type(response) == str
    assert f'Venda realizada com sucesso: {qty}' in response
