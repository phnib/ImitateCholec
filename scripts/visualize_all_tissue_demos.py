import os
import glob

import cv2

from ImitateCholec.utils.helper import get_valid_demo_start_end_indices


def generate_combined_video_all_demos(base_path, output_path, tissue_name, desired_camera_dir_names=["endo_psm2", "da_vinci_stereo_left", "da_vinci_stereo_right", "endo_psm1"]):
    
    print(f"Generating combined video for tissue: {tissue_name} with cameras: {desired_camera_dir_names}")
    
    # Define the final video path for each run
    final_video_path = os.path.join(output_path, f"all_demos_combined_{tissue_name}.mp4")
    
    # Create the parent directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # Define video writer
    out = None

    # Get the defined tissue dir
    tissue_dir_path = os.path.join(base_path, tissue_name)

    # Get and sort task dirs based on the number before the first "_"
    task_pattern = os.path.join(tissue_dir_path, '*_*')
    task_dirs = [task_dir for task_dir in glob.glob(task_pattern) if os.path.isdir(task_dir) and os.path.basename(task_dir).split('_')[0].isdigit()]
    task_dirs_sorted = sorted(task_dirs, key=lambda p: int(os.path.basename(p).split('_')[0]))
    
    if not task_dirs_sorted:
        print(f"No task dirs found for tissue {tissue_name} in {tissue_dir_path}")
        return

    # Iterate through each of the sorted task dirs
    for task_dir_path in task_dirs_sorted:
        print(f"Processing task: {os.path.basename(task_dir_path)}")
        
        # Get all demo dirs for that specific tissue and task
        demo_pattern = os.path.join(task_dir_path, '*-*')
        demo_dirs = [demo_dir for demo_dir in glob.glob(demo_pattern) if os.path.isdir(demo_dir)]
        if not demo_dirs:
            print(f"No demo dirs found for {task_dir_path}")
            continue

        demo_dirs_sorted = sorted(demo_dirs, key=lambda p: os.path.basename(p))
        for selected_demo_dir_path in demo_dirs_sorted:
            print(f"Processing demo: {os.path.basename(selected_demo_dir_path)}")
            
            new_demo_flag = True
            #  Get the start and end indices for the frames
            start, end = 0, get_valid_demo_start_end_indices(selected_demo_dir_path)[1]         
                
            # Process images for each frame index
            for frame_idx in range(start, end + 1):  
                images, widths = [], []
                
                for camera_dir in desired_camera_dir_names:
                    frame_idx_abs = abs(frame_idx)
                    img_path = os.path.join(selected_demo_dir_path, camera_dir, f"frame{str(frame_idx_abs).zfill(6)}.jpg")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            if camera_dir in ['da_vinci_stereo_left', 'da_vinci_stereo_right']:
                                if desired_camera_dir_names != ['da_vinci_stereo_left']:
                                    # Resize the image to 480 height
                                    height = 480
                                    width = int(img.shape[1] * (height / img.shape[0]))
                                    img = cv2.resize(img, (width, height))
                                
                            images.append(img)
                            widths.append(img.shape[1])
                        else:
                            raise ValueError(f"Image corrupt for {img_path}")
                    else:
                        print(f"Image not found for {img_path}")

                # Concatenate the images
                final_image = cv2.hconcat(images)
                
                
                # Add information about the task and demo on the left side of the video
                # Calculate text position for the task on the left
                task_text = f"Surgical Task: {os.path.basename(task_dir_path)}"
                font_scale = 0.7
                cv2.putText(final_image, task_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                # Calculate text position for the demo on the left (below the task)
                demo_text = f"Demo: {os.path.basename(selected_demo_dir_path)}"
                cv2.putText(final_image, demo_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

                # Write the final image to the video
                if out is None:
                    h, w, _ = final_image.shape
                    out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                out.write(final_image)
                if new_demo_flag:
                    # Hold the first frame for a short time
                    for _ in range(45):
                        out.write(final_image)
                    new_demo_flag = False

    # Release the video writer
    if out:
        out.release()
    print(f"Saved all concatenated demos video for tissue {tissue_name} in {final_video_path}")


if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')    
    
    tissue_names = ["tissue_31"]   
    data_collector_name = "data_collector_1" # data_collector_1 | data_collector_2  
    desired_camera_names = ["endo_psm2", "da_vinci_stereo_left", "da_vinci_stereo_right", "endo_psm1"]
    
    # Set the base and output path
    base_path = os.path.join(os.getenv("DATASET_PATH"), data_collector_name)
    output_path = os.path.join(os.getenv("OUTPUTS_PATH"), "AllTissueDemosVideos")

    # Generate the combined video
    for tissue_name in tissue_names:
        generate_combined_video_all_demos(base_path, output_path, tissue_name, desired_camera_names)
    print(f"All combined videos created")
    print(f"Output path: {output_path}")
