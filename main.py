from src.api import Client
#from src.simple_robot import SimpleRobot
from src.pycaret_robot import PycaretRobot

# config gerais
api = Client('dev')
minutos = 30 # trade a cada 1 min

""" Baseline
simple_robot = SimpleRobot(api)
simple_robot.train()
simple_robot.execute(minutos)
"""

# Best Model Selector
best_selector = PycaretRobot(api)
#best_selector.train()
best_selector.execute(minutos)
