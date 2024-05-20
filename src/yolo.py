from typing import Dict, List, Tuple
from ultralytics import YOLO


             
class YOLOFinetuning:
    def __init__(self, model_name:str, yolo_dataset_path:str, 
                 train_parameters:Dict=None, val_parameters:Dict=None, predict_parameters:Dict=None):
        self.model = YOLO(model_name)
        self.yolo_dataset_path = yolo_dataset_path
        
        self.training_results = None
        self.val_results = None
        
        self.train_parameters = train_parameters 
        self.val_parameters = val_parameters 
        self.predict_parameters = predict_parameters
        
    def train(self, parameters:Dict=None):
        parameters = self.train_parameters if self.train_parameters else parameters
        self.training_results = self.model.train(data=self.yolo_dataset_path, **parameters)
        
    def evaluate(self, parameters:Dict=None):
        parameters = self.train_parameters if self.train_parameters else parameters
        self.val_results = self.model.val(data=self.yolo_dataset_path, **parameters)
        
    def inference(self, frame, parameters:Dict, stream=False):
        parameters['stream'] = stream
        return self.model.predict(source=frame, **parameters) if not stream else self.model(frame, stream=True, **parameters)
    
    
        
    def hyperparameters_finetuning(self, optuna_hyperparameters:List[Tuple], metric_to_optimize='mAP_0.5', direction='maximize', n_trials=50):
        optuna = OptunaYoloHyperparamsFinetuning(self.yolo_dataset_path, self.model, optuna_hyperparameters)
        
        best_params, self.model, self.training_results, self.val_results = optuna.optuna_finetuning(metric_to_optimize=metric_to_optimize, 
                                                                                                    direction=direction, 
                                                                                                    n_trials=n_trials)
        return best_params, self.model, self.training_results, self.val_results


     
                   
    
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
        
        start_time = timer()
        # Set hyperparameters and train model
        self.model.train(data=self.dataset_path, **self._set_trial_params(trial))
        end_time = timer()
        training_time = f"{(end_time-start_time):.4f}"
        # Validate the model
        results = self.model.val(data=self.dataset_path)
        # Return the evaluation metric (e.g., mAP) for Optuna to optimize
        return results[f'metrics/{self.metric_to_optimize}'], training_time  # Adjust based on the actual result keys
    
    def _retrain_and_validate_model_best_params(self, best_params:Dict):
        # Set best hyperparameters and train model
        train_results = self.model.train(data=self.dataset_path, **best_params)
        val_results =  self.model.val(data=self.dataset_path)
        return train_results, val_results
        
    def optuna_finetuning(self, metric_to_optimize='mAP_0.5', direction=['maximize', 'minimize'], n_trials=50, custom_objective = None):
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

    
