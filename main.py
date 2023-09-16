from src.api import Client
from src.simple_robot import my_robot

api = Client('dev')
minutos = 10
my_robot(api, minutos)