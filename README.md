# Overview

The Fall Detection App is a tool designed to detect human falls from various sources including images, videos, YouTube links, and real-time streams. Leveraging the power of YOLOV8 (You Only Look Once) object detection models, this application ensures accurate and timely detection of falls, which is critical for ensuring safety in environments such as elder care facilities and workplaces.

## Features

- Multiple Input Sources: Supports images, image URLs, videos, YouTube links, and real-time streams.
- Streamlit Interface: A user-friendly web interface for ease of use.
- Real-Time Detection: Processes and detects falls in real-time streams.
- Hyperparameter Tuning: Utilizes Optuna for fine-tuning YOLO hyperparameters to improve model performance.
- Configurable: Easily customizable parameters for training and inference.


## Installation and setup

### Install poetry
https://python-poetry.org/docs/

### Clone the repository

    git clone https://github.com/ValentinOzeel/Fall_Detection_App.git
    cd Fall_Detection_App

### Activate your virtual environment with your favorite environment manager such as venv or conda (or poetry will create one)

    poetry install

### Run the app (from Fall_Detection_App)

    streamlit run src/main.py

## App tests

https://github.com/ValentinOzeel/Fall_Dectection_App/assets/117592568/4fc2b593-f7c0-4bbd-9123-ea01a9539df7


## Finetunning the model

WARNING: pytorch was not included as a dependency, please install pytorch (https://pytorch.org/get-started/locally/) if you want to retrain the model. Before starting the training process, a print statement will indicate whether or not cuda is available on your machine for GPU acceleration.

- Easily configure training and validation parameters as well as Optuna hyperparameters for model tuning via the conf\config.yaml file. 
- To carry out hyperparameter finetunning, run the following:

        python src/yolo_optuna.py

- To carry out training, run the following:

        python src/yolo_training.py

## Training roadmap

Used dataset for training the model:  Fall Detection - v4 resized640_aug3x-ACCURATE (see data\FallDetection.v4-resized640_aug3x-accurate.yolov8\README.dataset.roboflow.txt)

- Started with a dummy hyperparameter tuning run (optimizing mAP50 and mAP50-95) on a subset of the whole dataset (20%) to reduce computing time. Performed 300 trials with model training during a single epoch to narrow the initial range of parameter values.

        - Trial 315 finished with values: [0.449438970574703, 0.17658981830795356] and parameters: {'batch': 4, 'lr0': 0.0026223881897466414, 'optimizer': 'RAdam', 'dropout': 0.18286270267335275, 'momentum': 0.8217599345004953, 'weight_decay': 1.0918479453079965e-05}.    

- Then hyperparameter tuning (optimizing mAP50 and mAP50-95) was performed on the whole dataset, with focus on RAdam optimizer, still with a single epoch. The goal was to identify an ideal starting point for further training/finetuning.

        - Trial 12 finished with values: [0.5449658962834142, 0.24268036501245566] and parameters: {'batch': 3, 'lr0': 0.0021015698201867134, 'dropout': 0.001557670489734042, 'freeze': 10, 'momentum': 0.9097303930670215, 'weight_decay': 3.222409029324563e-05}.  

- Final tuning was then carried out for 100 epochs with narrower hyperparameter windows. 

        - Trial 4 finished with values: [0.8431803258501873, 0.49493992200926773] and parameters: {'batch': 4, 'lr0': 0.00144615503351372, 'lrf': 2.8465259340498403e-06, 'dropout': 0.10478331509780686, 'freeze': 7, 'momentum': 0.9087559754382691, 'weight_decay': 0.00032459862370794976}. 

- Final training hyperparameters and training results can be found at results\final_training. mAP50/mAP50-95: 0.85700, 0.51372



## Room for future improvments

- App input validation with tools such as pydantic
- Use a more robust YOLOV8 model as here the YOLOV8n model (n for nano, the lightest and fastest but less performant YOLOV8 model overall) was finetuned and trained 
- Cloud deployment
