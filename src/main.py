import os
from typing import Dict, List, Tuple


from secondary_module import project_root_path, ConfigLoad, colorize, check_cuda_availability

from yolo import YOLOFinetuning, OptunaYoloHyperparamsFinetuning


if __name__ == "__main__":
    
    check_cuda_availability()
    
    yolo_dataset_config_path = os.path.join(project_root_path, 'conf', 'YOLO_dataset.yaml') 
    config_path = os.path.join(project_root_path, 'conf', 'config.yaml')        
    config_load = ConfigLoad(path=config_path)
    config = config_load.get_config()

    yolo_model = config['YOLO']['MODEL']
    train_parameters = config['YOLO']['TRAIN_PARAMS']
    val_parameters = config['YOLO']['VAL_PARAMS']
    predict_parameters = config['YOLO']['PREDICT_PARAMS']



    import torch
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(x)
    
    
    yolo_ft = YOLOFinetuning(yolo_model, yolo_dataset_config_path, 
                             train_parameters=train_parameters, val_parameters=val_parameters, predict_parameters=predict_parameters)

    yolo_ft.train()

    
    
#    
#    
#    # Load your trained YOLOv8 model
#    model_path = 'path/to/your/fine-tuned-model.pt'
#    model = YOLO(model_path)

#    
#    # Initialize and run the FallDetectApp
#    FallDetectApp(model)
#    
#
#
#save the filke and run ---- streamlit run app.py ----
#MODEL =
#
#USE OUR YOLO CLASS FOR THE PREDICTION (WE PASS OUR PARAMS)
#
#
#
#
#app = FallDetectApp(model)