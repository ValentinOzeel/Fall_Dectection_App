import os

from secondary_module import project_root_path, ConfigLoad, colorize, check_cuda_availability
from yolo import YOLOFinetuning

if __name__ == "__main__":
    
    check_cuda_availability()
    
    yolo_dataset_config_path = os.path.join(project_root_path, 'conf', 'YOLO_dataset.yaml') 
    config_path = os.path.join(project_root_path, 'conf', 'config.yaml')        
    config_load = ConfigLoad(path=config_path)
    config = config_load.get_config()

    yolo_model = config['YOLO']['MODEL']
    train_parameters = config['YOLO']['TRAIN_PARAMS']
    val_parameters = config['YOLO']['VAL_PARAMS']

    
    yolo_ft = YOLOFinetuning(project_root_path, yolo_model, yolo_dataset_config_path, 
                             train_parameters=train_parameters, val_parameters=val_parameters)

    yolo_ft.train()
    