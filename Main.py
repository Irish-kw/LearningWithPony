# YOLO GAME
import json
from Function import Main_function

with open(f'Config.json', 'r') as f:
    config = json.load(f)

Main = Main_function(config)

if __name__ == '__main__':
    Main.init()
