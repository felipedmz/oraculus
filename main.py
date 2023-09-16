from src.api import Client

api = Client('dev')

print("\n> Verificando criptos disponíveis")
print(api.available_cripto())
#
print("\n> Obtendo status da carteira")
print(api.status())
#
print("\n> Verificando histórico de todas as movimentações")
print(api.history())
#
print("\n> Obtendo cotação dos últimos 500 minutos (OHLC) de uma cripto")
print(api.cripto_quotation())
#
print("\n> Realizando COMPRA de cripto")
print(api.buy(0.2))
#
print("\n> Realizando VENDA cripto")
print(api.sell(0.2))

