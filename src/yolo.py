from typing import Dict, List, Tuple
from ultralytics import YOLO
from yolo_optuna import OptunaYoloHyperparamsFinetuning


             
class YOLOFinetuning:
    def __init__(self, project_root_path:str, model_name:str, yolo_dataset_path:str, 
                 train_parameters:Dict=None, val_parameters:Dict=None, predict_parameters:Dict=None):
        self.project_root_path = project_root_path
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.yolo_dataset_path = yolo_dataset_path
        
        self.training_results = None
        self.val_results = None
        
        self.train_parameters = train_parameters 
        self.val_parameters = val_parameters 
        self.predict_parameters = predict_parameters
        
    def train(self, parameters:Dict=None):
        parameters = self.train_parameters if self.train_parameters else parameters
        self.training_results = self.model.train(data=self.yolo_dataset_path, **parameters) if parameters else self.model.train(data=self.yolo_dataset_path)
        
    def evaluate(self, parameters:Dict=None):
        parameters = self.train_parameters if self.train_parameters else parameters
        self.val_results = self.model.val(data=self.yolo_dataset_path, **parameters) if parameters else self.model.val(data=self.yolo_dataset_path)
        
    def inference(self, frame, parameters:Dict, stream=False):
        parameters['stream'] = stream
        return self.model.predict(source=frame, **parameters)
    
    
        
    def hyperparameters_finetuning(self, experiment_name:str, optuna_hyperparameters:List[Tuple],
                                   frozen_hyperparameters:Dict=None,
                                   metric_to_optimize=['map50', 'map'], directions=['maximize', 'maximize'], 
                                   n_trials=50):
        optuna = OptunaYoloHyperparamsFinetuning(self.project_root_path, experiment_name, self.yolo_dataset_path, 
                                                 self.model_name, 
                                                 optuna_hyperparameters, frozen_hyperparameters=frozen_hyperparameters)
        
        self.best_trials = optuna.optuna_finetuning(metric_to_optimize=metric_to_optimize, 
                                               directions=directions, 
                                               n_trials=n_trials)
        return self.best_trials

