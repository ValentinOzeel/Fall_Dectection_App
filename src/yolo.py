import os
from typing import Dict, List, Tuple


from secondary_module import project_root_path, ConfigLoad, colorize

from ultralytics import YOLO



yolo_dataset_config = os.path.join(project_root_path, 'conf', 'YOLO_dataset.yaml') 
config_path = os.path.join(project_root_path, 'conf', 'config.yaml')        
config_load = ConfigLoad(path=config_path)
config = config_load.get_config()




yolo_model_name = config['YOLO']['MODEL']
parameters = config['YOLO']['PARAMS']




             
class YOLOFinetuning:
    def __init__(self, model_name:str, yolo_dataset_path:str):
        self.model = YOLO(model_name)
        self.yolo_dataset_path = yolo_dataset_path
        
        self.training_results = None
        self.val_results = None
        
    def train(self, parameters:Dict):
        self.training_results = self.model.train(data=self.yolo_dataset_path, **parameters)
        
    def evaluate(self):
        self.val_results = self.model.val()
        
    def hyperparameters_finetuning(self, optuna_hyperparameters:List[Tuple], metric_to_optimize='mAP_0.5', direction='maximize', n_trials=50):
        optuna = OptunaYoloHyperparamsFinetuning(self.yolo_dataset_path, self.model, optuna_hyperparameters)
        
        best_params, self.model, self.training_results, self.val_results = optuna.optuna_finetuning(metric_to_optimize=metric_to_optimize, 
                                                                                                    direction=direction, 
                                                                                                    n_trials=n_trials)
        return best_params, self.model, self.training_results, self.val_results

            
    def SAVE MODEL 
     
     
     
     
     
     




                   
    
class OptunaYoloHyperparamsFinetuning:
    def __init__(self, dataset_path, model, parameters:List[Tuple]):
        self.dataset_path = dataset_path
        self.model = model
        self.parameters = parameters
        
        self.trials_counter = 0
        
        
    def _set_trial_params(self, trial) -> Dict:
        optuna_parameters = {}
        for param_name, suggest_type, param_range, param_options in self.parameters:
            trial_value = getattr(trial, suggest_type)(param_name, *param_range, **param_options)
            optuna_parameters[param_name] = trial_value
        return optuna_parameters
        
        
        
    def _objective(self, trial):
        self.trials_counter += 1
        # Set hyperparameters and train model
        self.model.train(data=self.dataset_path, **self._set_trial_params(trial))
        # Validate the model
        results = self.model.val(data=self.dataset_path)
        # Return the evaluation metric (e.g., mAP) for Optuna to optimize
        return results[f'metrics/{self.metric_to_optimize}']  # Adjust based on the actual result keys
    
    def _retrain_and_validate_model_best_params(self, best_params:Dict):
        # Set best hyperparameters and train model
        train_results = self.model.train(data=self.dataset_path, **best_params)
        val_results =  self.model.val(data=self.dataset_path)
        return train_results, val_results
        
    def optuna_finetuning(self, metric_to_optimize='mAP_0.5', direction='maximize', n_trials=50, custom_objective = None):
        self.metric_to_optimize = metric_to_optimize
        objective = self._objective if custom_objective is None else custom_objective
        # Create a study and optimize the objective function
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        # Reset counter
        self.trials_counter = 0
    
        # Retrain model with best parameters
        train_results, val_results = self._retrain_and_validate_model_best_params(study.best_params)
        return study.best_params, self.model, train_results, val_results

    
