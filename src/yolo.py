from typing import Dict, List, Tuple
from ultralytics import YOLO
import optuna
from timeit import default_timer as timer
from secondary_module import colorize
             
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
    
    
        
    def hyperparameters_finetuning(self, optuna_hyperparameters:List[Tuple],
                                   frozen_hyperparameters:Dict=None,
                                   metric_to_optimize=['map50', 'map'], directions=['maximize', 'maximize', 'minimize'], 
                                   n_trials=50):
        optuna = OptunaYoloHyperparamsFinetuning(self.yolo_dataset_path, self.model, optuna_hyperparameters, frozen_hyperparameters=frozen_hyperparameters)
        
        self.best_trials = optuna.optuna_finetuning(metric_to_optimize=metric_to_optimize, 
                                               directions=directions, 
                                               n_trials=n_trials)
        return self.best_trials



            
                
class OptunaYoloHyperparamsFinetuning:
    def __init__(self, dataset_path, model, optuna_hyperparameters:List[Tuple], frozen_hyperparameters:Dict=None):
        self.dataset_path = dataset_path
        self.model = model
        self.optuna_hyperparameters = optuna_hyperparameters
        self.frozen_hyperparameters = frozen_hyperparameters
        
        self.trials_counter = 0
        
        
    def _set_trial_params(self, trial) -> Dict:
        optuna_parameters = {}
        for param_name, suggest_type, param_range in self.optuna_hyperparameters:
            if 'categorical' in suggest_type:
                trial_value = getattr(trial, suggest_type)(param_name, param_range)
            else:
                trial_value = getattr(trial, suggest_type)(param_name, *param_range)
            optuna_parameters[param_name] = trial_value
        if self.frozen_hyperparameters:
            for key, value in self.frozen_hyperparameters.items():
                optuna_parameters[key] = value
        return optuna_parameters
        
        
        
    def _objective(self, trial):
        self.trials_counter += 1
        print('Optuna trial number ', colorize(str(self.trials_counter), 'RED'))
        start_time = timer()
        # Set hyperparameters and train model
        results = self.model.train(data=self.dataset_path, **self._set_trial_params(trial))
        end_time = timer()
        training_time = f"{(end_time-start_time):.4f}"
        # Validate the model
        #results = self.model.val(data=self.dataset_path)
        # Return the evaluation metric (e.g., mAP) for Optuna to optimize        
        
        metrics_to_optimize = [float(getattr(results.box, metric)) for metric in self.metric_to_optimize]
        metrics_to_optimize.append(float(training_time))
        return metrics_to_optimize  # Adjust based on the actual result keys
    
    def _retrain_and_validate_model_best_params(self, best_params:Dict):
        # Set best hyperparameters and train model
        if best_params.get('save_dir'):
            best_params['save_dir'] = "runs\\detect\\optuna_best"
        train_results = self.model.train(data=self.dataset_path, **best_params)
        val_results =  self.model.val(data=self.dataset_path)
        return train_results, val_results
        
    def optuna_finetuning(self, metric_to_optimize=['map50', 'map'], directions=['maximize', 'maximize', 'minimize'], n_trials=50, custom_objective = None):
        print('\n\nOptuna optimization begins...\n\n')
        self.metric_to_optimize = metric_to_optimize
        objective = self._objective if custom_objective is None else custom_objective
        # Create a study and optimize the objective function
        
        study = optuna.create_study(directions=directions)
        study.optimize(objective, n_trials=n_trials)
        # Reset counter
        self.trials_counter = 0

        # Retrain model with best parameters
   #     train_results, val_results = self._retrain_and_validate_model_best_params(study.best_params)
        print(colorize('\n\nOptuna best trials:', 'RED'))
        metric_to_optimize.append('training time')
        for trial in study.best_trials:
            print(colorize('\n\n----------------------------------------------\n', 'LIGHTRED_EX'), trial.number)
            print(colorize('TRIAL NUMBER: ', 'LIGHTGREEN_EX'), trial.number)
            print(colorize('HYPERPARAMETERS: ', 'LIGHTBLUE_EX'), trial.params)
            print(colorize(f'VALUES ({metric_to_optimize}): ', 'LIGHTMAGENTA_EX'), trial.values)
        return study.best_trials
     #   return study.best_params, self.model

    
