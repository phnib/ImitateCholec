import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import pandas as pd

from ImitateCholec.utils.rectify_wrist_cam_img import rectify_wrist_cam_image


class ImitationLearningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        all_splits_data_collector_tissue_dict,
        cameras_to_use=['endo_psm2', 'da_vinci_stereo_left', 'da_vinci_stereo_right', 'endo_psm1'],
        input_transforms=None,
        stride=1,
        chunk_size=100,
        apply_rectificiation_flag=False, # Flag to apply rectification on endo_psm1 camera images for tissue 3 to 9
    ):
        super().__init__()
        
        # Set attributes
        self.dataset_dir = dataset_dir
        self.all_splits_data_collector_tissue_dict = all_splits_data_collector_tissue_dict
        self.cameras_to_use = cameras_to_use
        self.input_transforms = input_transforms
        self.stride = stride 
        self.chunk_size = chunk_size
        self.apply_rectificiation_flag = apply_rectificiation_flag  

        # Get all images and kinematics paths (and the start and end indices for kinematics wrt to chunk size) 
        self.image_camera_inputs_paths_list, self.current_surgical_task_indices, self.kinematics_path_list, self.current_kinematics_idx_list = [], [], [], []
        self.sample_tissue_mapping = []  # Mapping from sample index to tissue name
        for data_collector, tissue_list in self.all_splits_data_collector_tissue_dict.items():
            for tissue in tissue_list:
                tissue_dir_path = os.path.join(self.dataset_dir, data_collector, tissue)
                task_dirs = os.listdir(tissue_dir_path)
                task_dirs = sorted(task_dirs, key=lambda p: int(p.split('_')[0]))  # Sort by task number
                for task_dir in task_dirs:
                    # Load all demo directories for the current task
                    task_dir_path = os.path.join(tissue_dir_path, task_dir)
                    demo_dirs = os.listdir(task_dir_path)
                    for demo_dir in demo_dirs:                        
                        # Get all image paths for the current demo (for all selected cameras)
                        demo_dir_path = os.path.join(task_dir_path, demo_dir)
                        left_img_dir_path = os.path.join(demo_dir_path, "da_vinci_stereo_left")  
                        frame_names_sorted = sorted(
                            [frame_name for frame_name in os.listdir(left_img_dir_path) if frame_name.endswith('.jpg')],
                            key=lambda x: int(os.path.basename(x).split('.')[0].replace('frame', ''))
                        )
                        frame_names_sorted = frame_names_sorted[::self.stride]  # Apply stride to the frames
                        # Go over each frame and construct the image paths for all cameras
                        for idx, frame_name in enumerate(frame_names_sorted):
                            self.current_surgical_task_indices.append(task_dir)  # Store the current surgical task
                            
                            # Construct the image paths for all cameras
                            cam_images_list = []
                            for camera_dir_name in self.cameras_to_use:
                                # Construct the camera directory path
                                camera_dir_path = os.path.join(demo_dir_path, camera_dir_name)
                                cam_images_list.append(os.path.join(camera_dir_path, frame_name))
                            self.image_camera_inputs_paths_list.append(cam_images_list)       
                        
                            # Load kinematics path 
                            kinematics_path = os.path.join(demo_dir_path, "kinematics.csv")
                            self.kinematics_path_list.append(kinematics_path)
                            # Store current kinematics idx
                            current_idx = idx * self.stride  # Current index relative to the stride
                            self.current_kinematics_idx_list.append(current_idx)
                            # Store tissue name mapping 
                            self.sample_tissue_mapping.append(tissue)
                        
    
    def __len__(self):
        return len(self.image_camera_inputs_paths_list)


    def __getitem__(self, idx):
        # Get the image paths and label index
        camera_image_paths_list = self.image_camera_inputs_paths_list[idx]
        surgical_task_idx = self.current_surgical_task_indices[idx]
        tissue_name = self.sample_tissue_mapping[idx]
        
        # Load kinematics 
        kinematics_path = self.kinematics_path_list[idx]
        current_kinematics_idx = self.current_kinematics_idx_list[idx]
        kinematics_chunk = self.load_kinematics_chunk(kinematics_path, current_kinematics_idx)       
        
        # Load raw images for all cameras
        camera_images = []
        for camera_name, image_path in zip(cameras_to_use, camera_image_paths_list):
            image = Image.open(image_path).convert('RGB')
            
            # Apply rectification transform on endo_psm1 camera images for tissue 3 to 9
            if self.apply_rectificiation_flag and camera_name == "endo_psm1" and tissue_name in [f"tissue_{idx}" for idx in range(3, 10)]:
                image = rectify_wrist_cam_image(image, angle=-52.0, shift_x=10, shift_y=0)
            
            # Apply transforms
            if self.input_transforms is not None:
                image = self.input_transforms(image)
            camera_images.append(image)
        
        return camera_images, surgical_task_idx, kinematics_chunk
    
    
    def load_kinematics_chunk(self, kinematics_path, current_kinematics_idx):
        """
        Load kinematics data from the specified path and return the relevant chunk based on start and end indices.
        """
        
        # Define the columns to load from the kinematics data
        kinematics_to_load = ['psm1_pose.position.x', 'psm1_pose.position.y', 'psm1_pose.position.z',
                              'psm1_pose.orientation.x', 'psm1_pose.orientation.y', 'psm1_pose.orientation.z', 'psm1_pose.orientation.w',
                              'psm1_jaw',
                              'psm2_pose.position.x', 'psm2_pose.position.y', 'psm2_pose.position.z',
                              'psm2_pose.orientation.x', 'psm2_pose.orientation.y', 'psm2_pose.orientation.z', 'psm2_pose.orientation.w',
                              'psm2_jaw']
        
        # Load the kinematics data 
        kinematics_data = pd.read_csv(kinematics_path)
        kinematics_data = kinematics_data[kinematics_to_load]

        # Transform orientation columns from quaternion to Euler angles
        kinematics_data = self.extract_euler_angles_in_df(kinematics_data)

        # Extract the relevant chunk of kinematics data
        start_idx, end_idx = current_kinematics_idx + 1, current_kinematics_idx + self.chunk_size + 1
        valid_end =  min(end_idx, len(kinematics_data) - 1)  # Ensure end index is within bounds
        kinematics_chunk = kinematics_data.iloc[start_idx:valid_end]

        # Repeat last kinematics if the chunk is smaller than chunk_size
        if len(kinematics_chunk) < self.chunk_size:
            last_row = kinematics_chunk.iloc[-1].values
            num_repeats = self.chunk_size - len(kinematics_chunk)
            repeated_rows = np.tile(last_row, (num_repeats, 1))
            kinematics_chunk = pd.concat([kinematics_chunk, pd.DataFrame(repeated_rows, columns=kinematics_chunk.columns)], ignore_index=True)

        return kinematics_chunk  # Convert to numpy array
        
        
    @staticmethod
    def extract_euler_angles_in_df(kinematics_data):
        """Convert quaternion orientations to Euler angles (roll, pitch, yaw)"""
        
        # PSM1 quaternion to Euler conversion
        qx1, qy1, qz1, qw1 = (kinematics_data['psm1_pose.orientation.x'], 
                              kinematics_data['psm1_pose.orientation.y'],
                              kinematics_data['psm1_pose.orientation.z'], 
                              kinematics_data['psm1_pose.orientation.w'])
        
        # Roll (x-axis rotation)
        roll1 = np.arctan2(2 * (qw1 * qx1 + qy1 * qz1), 1 - 2 * (qx1**2 + qy1**2))
        # Pitch (y-axis rotation)
        sin_pitch1 = 2 * (qw1 * qy1 - qz1 * qx1)
        pitch1 = np.where(np.abs(sin_pitch1) >= 1, np.copysign(np.pi / 2, sin_pitch1), np.arcsin(sin_pitch1))
        # Yaw (z-axis rotation)
        yaw1 = np.arctan2(2 * (qw1 * qz1 + qx1 * qy1), 1 - 2 * (qy1**2 + qz1**2))
        
        # PSM2 quaternion to Euler conversion
        qx2, qy2, qz2, qw2 = (kinematics_data['psm2_pose.orientation.x'], 
                              kinematics_data['psm2_pose.orientation.y'],
                              kinematics_data['psm2_pose.orientation.z'], 
                              kinematics_data['psm2_pose.orientation.w'])
        
        # Roll (x-axis rotation)
        roll2 = np.arctan2(2 * (qw2 * qx2 + qy2 * qz2), 1 - 2 * (qx2**2 + qy2**2))
        # Pitch (y-axis rotation)
        sin_pitch2 = 2 * (qw2 * qy2 - qz2 * qx2)
        pitch2 = np.where(np.abs(sin_pitch2) >= 1, np.copysign(np.pi / 2, sin_pitch2), np.arcsin(sin_pitch2))
        # Yaw (z-axis rotation)
        yaw2 = np.arctan2(2 * (qw2 * qz2 + qx2 * qy2), 1 - 2 * (qy2**2 + qz2**2))
        
        # Replace quaternion columns with Euler angles
        kinematics_data = kinematics_data.drop(columns=[
            'psm1_pose.orientation.x', 'psm1_pose.orientation.y', 'psm1_pose.orientation.z', 'psm1_pose.orientation.w',
            'psm2_pose.orientation.x', 'psm2_pose.orientation.y', 'psm2_pose.orientation.z', 'psm2_pose.orientation.w'
        ])
        
        # Add Euler angle columns
        kinematics_data['psm1_pose.orientation.roll'] = roll1
        kinematics_data['psm1_pose.orientation.pitch'] = pitch1
        kinematics_data['psm1_pose.orientation.yaw'] = yaw1
        kinematics_data['psm2_pose.orientation.roll'] = roll2
        kinematics_data['psm2_pose.orientation.pitch'] = pitch2
        kinematics_data['psm2_pose.orientation.yaw'] = yaw2
        
        return kinematics_data
        
    
if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')   

    from ImitateCholec.utils.transforms import get_transforms, destandardize
    from ImitateCholec.utils.visualizations_3d_position_orientation import plot_psm_position_trajectory

    # Set all parameters
    data_collector_tissue_dict = {
        "data_collector_1": [f"tissue_{i}" for i in range(1, 35) if i not in [24, 25, 26]], 
        "data_collector_2": [f"tissue_{i}" for i in [24, 25, 26]],
    }
    image_dim=(224, 224)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet mean and std
    apply_augmentations_on_train = False  # No augmentations for this dataset
    input_transforms = get_transforms(image_size=(224, 224), mean=mean, std=std,
                                      apply_augmentations_on_train=apply_augmentations_on_train)
    cameras_to_use=['endo_psm2', 'da_vinci_stereo_left', 'da_vinci_stereo_right', 'endo_psm1']
    dataset_dir = os.getenv("DATASET_PATH")
    stride = 20
    chunk_size = 100 # Future points to generate
    apply_rectificiation_flag = True

    # Create the dataset
    chole_dataset = ImitationLearningDataset(
        dataset_dir=dataset_dir,
        all_splits_data_collector_tissue_dict=data_collector_tissue_dict,
        input_transforms=input_transforms,
        cameras_to_use=cameras_to_use,
        stride=stride,
        chunk_size=chunk_size,
        apply_rectificiation_flag=apply_rectificiation_flag
    )

    # Create combined visualization with camera images and trajectory plot
    idx = np.random.randint(0, len(chole_dataset))
    camera_images_list, surgical_task_idx, kinematics_chunk = chole_dataset[idx]
    
    # Filter kinematics_chunk to only include position columns
    position_columns = [col for col in kinematics_chunk.columns if 'position' in col]
    kinematics_positions = kinematics_chunk[position_columns]
    
    # Create combined plot - adjust figure size based on number of cameras
    num_cameras = len(camera_images_list)
    fig = plt.figure(figsize=(6 * (num_cameras + 1), 8))  # +1 for trajectory plot
    
    # Horizontal layout: camera images side by side, then trajectory plot on the right
    gs = fig.add_gridspec(1, num_cameras + 1, width_ratios=[1] * num_cameras + [1.2])  # Slightly wider for 3D plot
    
    # Left side: Stack camera images horizontally
    for i, camera_img in enumerate(camera_images_list):
        ax_img = fig.add_subplot(gs[0, i])
        img_destandardized = destandardize(camera_img, mean, std)
        ax_img.imshow(img_destandardized.permute(1, 2, 0))
        ax_img.set_title(f"Camera {chole_dataset.cameras_to_use[i]} - Task: {surgical_task_idx}")
        ax_img.axis('off')
    
    # Right side: Trajectory plot in the last column
    ax_traj = fig.add_subplot(gs[0, -1], projection='3d')
    
    # Plot trajectory from kinematics_chunk
    plot_psm_position_trajectory(
        kinematics_chunk=kinematics_positions,
        unit_factor=100,
        x_offset=0,
        y_offset=2,
        z_offset=0,
        kinematics_step_size=1,
        display_title_flag=True,
        ax=ax_traj,
        fig=fig
    )
    
    plt.tight_layout()
    
    outputs_dir = os.path.join(os.getenv("OUTPUTS_PATH"), "DatasetExamples")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_path = os.path.join(outputs_dir, "imitation_learning_dataset_example.jpg")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Combined visualization saved as '{output_path}'")
    print(f"Sample shows task: {surgical_task_idx}, {num_cameras} camera views")
    print(f"Camera order: {chole_dataset.cameras_to_use}")
    print(f"Kinematics chunk shape: {kinematics_chunk.shape}")
    plt.close()
