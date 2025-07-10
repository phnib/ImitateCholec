import random
import os
import glob
from collections import deque
from datetime import datetime

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
    
from ImitateCholec.utils.helper import get_valid_demo_start_end_indices
from ImitateCholec.utils.visualizations_3d_position_orientation import quaternion_to_rotation_matrix, visualize_pose_reconstruction


def generate_video_entire_clipping_cutting_phase(base_path, output_path, timestamp, tissue_name="tissue_10", start_surgical_task=1, end_surgical_task=17, kinematics_vis_step_size=10, history_length=10,
                          coord_frame_vis_flag=True, desired_camera_dir_names=["endo_psm2", "da_vinci_stereo_left", "endo_psm1"], x_limits=[-5, 2], y_limits=[-2.5, 2.5], z_limits=[4, 8], print_jaw_values_flag=True):
   
    print(f"Creating video for tissue: {tissue_name}, from surgical task {start_surgical_task} to {end_surgical_task}")
   
    # Create the parent directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Variables for scaling and translating the kinematics for better visualization
    unit_factor = 100 # E.g. 100 for cm, 1000 for mm
    x_offset, y_offset, z_offset = 0, 2, 0 
    
    # ---------
    
    # Define the final video path for each run
    final_video_path = os.path.join(output_path, f"{tissue_name}_entire_clipping_cutting_phase_{timestamp}.mp4")
    
    # Define video writer
    out = None

    # Get the defined tissue dir
    tissue_dir_path = os.path.join(base_path, tissue_name)

    # Create the full kinematics episode dataframe
    full_kinematics_episode = pd.DataFrame()

    # Initialize history buffers with a maximum length of 'history_length'
    psm1_history = deque(maxlen=history_length)
    psm2_history = deque(maxlen=history_length)

    # Create a Matplotlib figure and axes for 3D plotting
    fig_3d_plot = plt.figure(figsize=(6, 6))
    ax_3d_plot = fig_3d_plot.add_subplot(111, projection='3d')

    frame_counter = 0

    # Iterate through each of surgical_task dirs
    for surgical_task_idx in range(start_surgical_task, end_surgical_task + 1):
        # Get the current surgical_task dir
        surgical_task_dir_start = f"{surgical_task_idx}_"
        surgical_task_dirs = [dir for dir in os.listdir(tissue_dir_path) if dir.startswith(surgical_task_dir_start) and "recovery" not in dir]
        if not surgical_task_dirs:
            raise ValueError(f"No dir found for surgical_task index {surgical_task_idx}")
        # Select the first matching surgical_task dir (or adjust the logic as needed)
        surgical_task_dir_path = os.path.join(tissue_dir_path, surgical_task_dirs[0])
                
        # Get all demo dirs for that specific tissue and surgical_task
        demo_dir_name_pattern = os.path.join(surgical_task_dir_path, '*-*')
        demo_dirs = [dir for dir in glob.glob(demo_dir_name_pattern) if os.path.isdir(dir)]
        if not demo_dirs:
            continue
        selected_demo_dir_path = random.choice(demo_dirs)
        
        # Get number of frames from the left image directory
        left_img_dir_path = os.path.join(selected_demo_dir_path, "da_vinci_stereo_left")
        if not os.path.exists(left_img_dir_path):
            print(f"No left image directory found for {selected_demo_dir_path}")
            continue
        start, end, *_ = get_valid_demo_start_end_indices(selected_demo_dir_path)
        
        # Get number of frames from the kinematics csv file
        kinematics_csv_path = os.path.join(selected_demo_dir_path, 'kinematics.csv')
        if not os.path.exists(kinematics_csv_path):
            print(f"No kinematics csv file found for {selected_demo_dir_path}")
            continue # Skip if no kinematics csv file found
        demo_kinematics = pd.read_csv(kinematics_csv_path)
        valid_demo_kinematics = demo_kinematics.iloc[start:end + 1]
        # Concatenate the kinematics data
        full_kinematics_episode = pd.concat([full_kinematics_episode, valid_demo_kinematics], ignore_index=True)            
        
        # Process images for each frame index
        for frame_idx in range(start, end + 1):
            
            if frame_counter % kinematics_vis_step_size == 0:
                kinematics_row = full_kinematics_episode.iloc[frame_counter]

                # Extract positions and orientations for PSM1 and PSM2
                # For PSM2 (left instrument)
                psm2_position = np.array([
                    kinematics_row['psm2_pose.position.x']*unit_factor - x_offset,
                    kinematics_row['psm2_pose.position.y']*unit_factor - y_offset,
                    kinematics_row['psm2_pose.position.z']*unit_factor - z_offset
                ], dtype=np.float32)
                psm2_orientation_x = kinematics_row['psm2_pose.orientation.x']
                psm2_orientation_y = kinematics_row['psm2_pose.orientation.y']
                psm2_orientation_z = kinematics_row['psm2_pose.orientation.z']
                psm2_orientation_w = kinematics_row['psm2_pose.orientation.w']
                psm2_rotation_matrix = quaternion_to_rotation_matrix(psm2_orientation_x, psm2_orientation_y, psm2_orientation_z, psm2_orientation_w)
                psm2_jaw = kinematics_row['psm2_jaw']

                # For PSM1 (right instrument)
                psm1_position = np.array([
                    kinematics_row['psm1_pose.position.x']*unit_factor + x_offset,
                    kinematics_row['psm1_pose.position.y']*unit_factor + y_offset,
                    kinematics_row['psm1_pose.position.z']*unit_factor + z_offset
                ], dtype=np.float32)
                psm1_orientation_x = kinematics_row['psm1_pose.orientation.x']
                psm1_orientation_y = kinematics_row['psm1_pose.orientation.y']
                psm1_orientation_z = kinematics_row['psm1_pose.orientation.z']
                psm1_orientation_w = kinematics_row['psm1_pose.orientation.w']
                psm1_rotation_matrix = quaternion_to_rotation_matrix(psm1_orientation_x, psm1_orientation_y, psm1_orientation_z, psm1_orientation_w)
                psm1_jaw = kinematics_row['psm1_jaw']

                # Append to history buffers with jaw values
                psm2_history.append((psm2_position, psm2_rotation_matrix, psm2_jaw))
                psm1_history.append((psm1_position, psm1_rotation_matrix, psm1_jaw))
                
                # Generate visualization image
                coord_frame_length = 0.8
                elev, azim = 30, 20
                visualization_image = visualize_pose_reconstruction(
                    psm2_history=psm2_history,
                    psm1_history=psm1_history,
                    fig_3d_plot=fig_3d_plot,
                    ax_3d_plot=ax_3d_plot,
                    elev=elev,
                    azim=azim,
                    x_range=x_limits,
                    y_range=y_limits,
                    z_range=z_limits,
                    coord_frame_length=coord_frame_length,
                    coord_frame_vis_flag=coord_frame_vis_flag,
                    print_jaw_values_flag=print_jaw_values_flag
                )
            
            images, widths = [], []
            # Get image for both wrist cameras and left (and maybe right) image from stereo camera 
            for camera_name in desired_camera_dir_names:
                img_path = os.path.join(selected_demo_dir_path, camera_name, f"frame{str(frame_idx).zfill(6)}.jpg")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        if camera_name == 'da_vinci_stereo_left' or camera_name == 'da_vinci_stereo_right':
                            height = 480
                            width = int(img.shape[1] * (height / img.shape[0]))
                            img = cv2.resize(img, (width, height))
                        images.append(img)
                        widths.append(img.shape[1])
                    else:
                        raise ValueError(f"Image corrupt for {img_path}")
                else:
                    raise ValueError(f"Image not found for {img_path}")
            if not images:
                raise Exception(f"No images found for frame {frame_idx}. Skipping.")

            # Resize visualization image to match the height of the other images
            vis_height = visualization_image.shape[0]
            vis_width = visualization_image.shape[1]
            target_height = images[0].shape[0]  # Assuming all images have the same height
            new_vis_width = int(vis_width * (target_height / vis_height))
            visualization_image_resized = cv2.resize(visualization_image, (new_vis_width, target_height))

            # Concatenate the images (original images on the left, visualization on the far right)
            final_image = cv2.hconcat(images + [visualization_image_resized])                

            # Calculate text position to center over the left_img_dir image
            text = f"Directory: {os.path.basename(surgical_task_dir_path)}"
            if len(widths) == 1:
                text_position_x = widths[0] // 2 - 325
            else:
                text_position_x = widths[0] + (widths[1] // 2) - 325  # Center text on the second image
            cv2.putText(final_image, text, (text_position_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the final image to the video
            if out is None:
                h, w, _ = final_image.shape
                out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            out.write(final_image)
            
            frame_counter += 1

    # ----------

    # Release the video writer
    if out:
        out.release()
        
    print(f"Saved video for surgical task {surgical_task_idx} at {final_video_path}")


if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')

    # Define the parameters for the video generation
    start_surgical_task, end_surgical_task = 1, 17 # From min surgical_task 1 to max surgical_task 17
    dataset_name = "data_collector_1"
    tissue_name = "tissue_31"  # tissue_1 has no continuous surgical_tasks, so we are using >= tissue_2
    desired_camera_dir_names = ["da_vinci_stereo_left"] 
    kinematics_vis_step_size = 6
    coord_frame_vis_flag = True
    history_length = 6
    print_jaw_values_flag = True
    
    # Set the base and output path
    base_path = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_path = os.path.join(os.getenv("OUTPUTS_PATH"), "EntireClippingCuttingPhaseVideos")
    
    # Generate the combined video
    generate_video_entire_clipping_cutting_phase(
        base_path=base_path,
        output_path=output_path,
        timestamp=timestamp,
        tissue_name=tissue_name,
        start_surgical_task=start_surgical_task,
        end_surgical_task=end_surgical_task,
        kinematics_vis_step_size=kinematics_vis_step_size,
        history_length=history_length,
        coord_frame_vis_flag=coord_frame_vis_flag,
        desired_camera_dir_names=desired_camera_dir_names,
        print_jaw_values_flag=print_jaw_values_flag
    )