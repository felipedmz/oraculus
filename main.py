from src.api import Client
from src.simple_robot import SimpleRobot

# config gerais
api = Client('dev')
minutos = 10 # 1 trade a cada 1 min

# Baseline
simple_robot = SimpleRobot(api)
simple_robot.train()
simple_robot.execute(minutos) 
