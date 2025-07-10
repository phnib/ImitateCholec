import os
import json
import torch

import pandas as pd


def get_valid_demo_start_end_indices(demo_dir_path):
    # Load the start and end indices for the current demo as the valid range of the demo
    frame_files = os.listdir(os.path.join(demo_dir_path, "da_vinci_stereo_left")) 
    start, end = 0, len(frame_files) - 1 
    demo_num_frames_valid = end - start + 1
    
    return start, end, demo_num_frames_valid


def get_all_sorted_task_dirs():
    # Get all sorted tasks
    task_dir_names_sorted = [
        "1_grasping_gallbladder",
        "2_clipping_first_clip_left_tube",
        "3_going_back_first_clip_left_tube",
        "4_clipping_second_clip_left_tube",
        "5_going_back_second_clip_left_tube",
        "6_clipping_third_clip_left_tube",
        "7_going_back_third_clip_left_tube",
        "8_go_to_the_cutting_position_left_tube",
        "9_go_back_from_the_cut_left_tube",
        "10_clipping_first_clip_right_tube",
        "11_going_back_first_clip_right_tube",
        "12_clipping_second_clip_right_tube",
        "13_going_back_second_clip_right_tube",
        "14_clipping_third_clip_right_tube",
        "15_going_back_third_clip_right_tube",
        "16_go_to_the_cutting_position_right_tube",
        "17_go_back_from_the_cut_right_tube"
    ]
    return task_dir_names_sorted

def get_all_tasks_sorted_by_execution_order():
    """ Get all tasks sorted by their execution order """
    
    task_dir_names_sorted = get_all_sorted_task_dirs()
    
    tasks = []
    
    for task_dir in task_dir_names_sorted:
        # Remove the number prefix and underscore (e.g., "1_" from "1_grabbing_gallbladder")
        task_name = task_dir.split('_', 1)[1]  # Split on first underscore and take the second part
        
        # Replace underscores with spaces and capitalize each word
        task_name = task_name.replace('_', ' ').title()
        
        tasks.append(task_name)
    
    return tasks
    

def get_tissues_data_collector_name(dataset_path, tissue):
    """ Determine the dataset name based on the tissue"""
    if tissue in os.listdir(os.path.join(dataset_path, "data_collector_1")):
        return "data_collector_1"
    else:
        return "data_collector_2"
    
    
if __name__ == "__main__":
    get_all_tasks_sorted_by_execution_order()