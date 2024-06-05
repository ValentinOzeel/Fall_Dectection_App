import os
from typing import Dict, List, Tuple

from ultralytics import YOLO
from secondary_module import project_root_path, ConfigLoad, colorize, check_cuda_availability
from yolo import YOLOFinetuning
from streamlit_app import FallDetectApp

if __name__ == "__main__":
    
    # Load your trained YOLOv8 model
    model = YOLO(r'results\final_training\weights\best.pt')
    # Launch the app
    FallDetectApp(model)
