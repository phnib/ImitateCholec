import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import deque

# Import necessary functions from your utils (adjust the import paths as needed)
from ImitateCholec.utils.visualizations_3d_position_orientation import generate_video_vision_kinematics_single_demo


if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')
    
    # Define the parameters for the video generation
    data_collector_name = "data_collector_1" # data_collector_1 | data_collector_2 
    base_path = os.path.join(os.getenv("DATASET_PATH"), data_collector_name)
    selected_tissue = "tissue_31"
    selected_task = "4_clipping_second_clip_left_tube"
    selected_demo = "20240907-092213-005003"
    output_dir_path = os.path.join(os.getenv("OUTPUTS_PATH"), 'SingleDemoVideos')
    os.makedirs(output_dir_path, exist_ok=True)

    # Parameters for visualization
    kinematics_vis_step_size = 3
    history_length = 8
    coord_frame_vis_flag = True
    print_jaw_values_flag = True

    # Generate the combined video
    generate_video_vision_kinematics_single_demo(
        base_path=base_path,
        selected_tissue=selected_tissue,
        selected_task=selected_task,
        selected_demo=selected_demo,
        output_dir_path=output_dir_path,
        kinematics_vis_step_size=kinematics_vis_step_size,
        history_length=history_length,
        coord_frame_vis_flag=coord_frame_vis_flag,
        print_jaw_values_flag=print_jaw_values_flag,
    )