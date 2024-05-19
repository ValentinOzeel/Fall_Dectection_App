import os 
import yaml
from colorama import init, Fore, Back, Style
init() # Initialize Colorama to work on Windows



# Assuming data_exploration.py is in src\main.py
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root_path, 'conf', 'config.yml')


class ConfigLoad():
    def __init__(self, path=config_path):
        self.path = path
        with open(self.path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def get_config(self):
        return self.config
            
            
def colorize(to_print, color):
    return f"{getattr(Fore, color) + to_print + Style.RESET_ALL}"
