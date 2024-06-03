import os
from typing import List, Tuple, Dict
from timeit import default_timer as timer
from ultralytics import YOLO
import optuna
from secondary_module import colorize

        

class OptunaYoloHyperparamsFinetuning:
    def __init__(self, project_root, experiment_name, dataset_path, model_name, optuna_hyperparameters: List[Tuple], frozen_hyperparameters: Dict = None):
        self.project_root = project_root
        self.experiment_name = experiment_name
        self.dataset_path = dataset_path
        self.model_name = model_name
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
                
        optuna_parameters['project'] = ''.join([optuna_parameters['project'], '/', self.experiment_name])

        return optuna_parameters
        
    def _objective(self, trial):
        self.trials_counter += 1
        print('Optuna trial number: ', colorize(str(self.trials_counter), 'RED'))
        
        # Reinitialize the model for each trial
        model = YOLO(self.model_name)
        
        start_time = timer()
        # Set hyperparameters and train model
        results = model.train(data=self.dataset_path, name=f'trial_{self.trials_counter}', **self._set_trial_params(trial))
        end_time = timer()
        training_time = f"{(end_time - start_time):.4f}"
        
        # Collect the metrics to optimize
        metrics_to_optimize = [float(getattr(results.box, metric)) for metric in self.metric_to_optimize]
  #      metrics_to_optimize.append(float(training_time))
        return metrics_to_optimize  # Adjust based on the actual result keys
    
    def optuna_finetuning(self, metric_to_optimize=['map50', 'map'], directions=['maximize', 'maximize'], n_trials=50, custom_objective=None):
        print('\n\nOptuna optimization begins...\n\n')
        self.metric_to_optimize = metric_to_optimize
        objective = self._objective if custom_objective is None else custom_objective
        
        # Create a study and optimize the objective function
        study = optuna.create_study(directions=directions)
        study.optimize(objective, n_trials=n_trials)
        
        # Reset counter
        self.trials_counter = 0


        self._print_trials(study.trials, 'Results of all trials: ', 'LIGHTRED_EX')
        self._print_trials(study.best_trials, 'Results of all trials: ', 'LIGHTYELLOW_EX')
  
        
        return study.best_trials
    
    def _print_trials(self, trials, first_print:str, delimiter_color:str):
        
        print(colorize(f'\n\n{first_print}', 'RED'))

        for trial in trials:
            print(colorize('\n\n----------------------------------------------\n', delimiter_color.upper()))
            print(colorize('TRIAL NUMBER: ', 'LIGHTGREEN_EX'), trial.number)
            print(colorize('HYPERPARAMETERS: ', 'LIGHTBLUE_EX'), trial.params)
            print(colorize(f'VALUES ({self.metric_to_optimize}): ', 'LIGHTMAGENTA_EX'), trial.values)