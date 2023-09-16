from flask import Flask, request
from src.simple_robot import my_robot
from src.api import Client

app = Flask(__name__)

@app.route("/")
def index():
    return "Robo Cripto"

@app.route('/wakeup', methods=["POST"])
def wakeup():
    tempo = int(request.form.get("time"))
    api = Client('dev')
    my_robot(tempo, api)
    
app.run()