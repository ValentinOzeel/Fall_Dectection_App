import os
from typing import Dict, List, Tuple

from ultralytics import YOLO
from secondary_module import project_root_path, ConfigLoad, colorize, check_cuda_availability
from yolo import YOLOFinetuning
from streamlit_app import FallDetectApp

if __name__ == "__main__":
    
    check_cuda_availability()
    
#    yolo_dataset_config_path = os.path.join(project_root_path, 'conf', 'YOLO_dataset.yaml') 
#    config_path = os.path.join(project_root_path, 'conf', 'config.yaml')        
#    config_load = ConfigLoad(path=config_path)
#    config = config_load.get_config()

#    yolo_model = config['YOLO']['MODEL']
#    train_parameters = config['YOLO']['TRAIN_PARAMS']
#    val_parameters = config['YOLO']['VAL_PARAMS']
#    predict_parameters = config['YOLO']['PREDICT_PARAMS']

#    optuna_hyperparameters = config['YOLO']['OPTUNA_PARAMS']
#    frozen_hyperparameters = config['YOLO']['OPTUNA_FROZEN_PARAMS']

    
    
#    yolo_ft = YOLOFinetuning(project_root_path, yolo_model, yolo_dataset_config_path, 
#                             train_parameters=train_parameters, val_parameters=val_parameters, predict_parameters=predict_parameters)

#    yolo_ft.train()
#    best_trials = yolo_ft.hyperparameters_finetuning('batch-4_100epoch', optuna_hyperparameters, frozen_hyperparameters=frozen_hyperparameters, n_trials=5)

    
    
    # Load your trained YOLOv8 model
    model = YOLO(r'results\final_training\weights\best.pt')
    # Launch the app
    FallDetectApp(model)
