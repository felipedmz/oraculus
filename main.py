from src.api import Client
from src.simple_robot import SimpleRobot

# config gerais
api = Client('dev')
minutos = 10 # 1 trade a cada 1 min

# Baseline
simple_robot = SimpleRobot(api)
simple_robot.train()
simple_robot.execute(minutos) 

""" Estrategia a ser seguida

grupo_robot = GroupRobot(api)
grupo_robot.calcula_preco()
    # -- train() -> preco previsto / arima
grupo_robot.calcula_p_value()
    # -- train() -> quanto de certeza eu tenho do preco de cima / cnn
grupo_robot.trade()
    # self.preco +20
    # self.p_value 96%
    # if tenho bitcoin
        # acao = buy
    # else 
        # nao existem fundos suficientes / stop
"""
