import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ImitateCholec.utils.helper import get_valid_demo_start_end_indices


def visualize_task_trajectory_overlay(task_dir_path, selected_tissue, selected_task_idx, display_legend_flag=True, display_title_flag=True, recovery_flag=False):
    """
    Visualizes the PSM1 and PSM2 trajectories for a given task by overlaying them in a 3D plot.
    
    Parameters:
    - task_dir_path: Path to the task dir containing demonstration data.
    - selected_tissue: Tissue name to be displayed in the title.
    - selected_task_idx: Index of the task to visualize.
    - display_legend_flag: Whether to display the legend on the plot.
    - display_title_flag: Whether to display the title on the plot.
    - recovery_flag: Whether to visualize recovery tasks.
    """

    # Init the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Init the total trajectory length variables
    psm1_trajectory_lengths = []
    psm2_trajectory_lengths = []
    total_trajectory_lengths = []

    # Go over all demos and overlay the trajectories
    for demo_idx, selected_demo_dir in enumerate(os.listdir(task_dir_path)):
        demo_dir_path = os.path.join(task_dir_path, selected_demo_dir)   
        kinematics_csv_path = os.path.join(demo_dir_path, 'kinematics.csv')
        demo_kinematics = pd.read_csv(kinematics_csv_path)

        # Get valid start and end positions
        start, end, *_ = get_valid_demo_start_end_indices(demo_dir_path)

        unit_factor = 100 # E.g. 100 for cm, 1000 for mm
        x_offset, y_offset, z_offset = 0, 2, 0 
        psm1_position = np.array([
        demo_kinematics['psm1_pose.position.x'][start:end+1]*unit_factor + x_offset,
        demo_kinematics['psm1_pose.position.y'][start:end+1]*unit_factor + y_offset,
        demo_kinematics['psm1_pose.position.z'][start:end+1]*unit_factor + z_offset
        ], dtype=np.float32)
        psm2_position = np.array([
        demo_kinematics['psm2_pose.position.x'][start:end+1]*unit_factor - x_offset,
        demo_kinematics['psm2_pose.position.y'][start:end+1]*unit_factor - y_offset,
        demo_kinematics['psm2_pose.position.z'][start:end+1]*unit_factor - z_offset
        ], dtype=np.float32)

        # Plot psm1 positions as scatter points in blue with lines connecting them
        start_point_size = 15
        ax.scatter(psm1_position[0, 0], psm1_position[1, 0], psm1_position[2, 0], c='blue', label='psm1 position', s=start_point_size)
        ax.scatter(psm1_position[0, 1:], psm1_position[1, 1:], psm1_position[2, 1:], c='blue', s=1)
        ax.plot(psm1_position[0], psm1_position[1], psm1_position[2], c='blue', linewidth=0.5)

        # Plot psm2 positions as scatter points in red with lines connecting them
        ax.scatter(psm2_position[0, 0], psm2_position[1, 0], psm2_position[2, 0], c='red', label='psm2 position', s=start_point_size)
        ax.scatter(psm2_position[0, 1:], psm2_position[1, 1:], psm2_position[2, 1:], c='red', s=1)
        ax.plot(psm2_position[0], psm2_position[1], psm2_position[2], c='red', linewidth=0.5)

        trajectory_length_psm2 = np.sum(np.linalg.norm(np.diff(psm2_position, axis=1), axis=0))
        trajectory_length_psm1 = np.sum(np.linalg.norm(np.diff(psm1_position, axis=1), axis=0))
        total_trajectory_length = trajectory_length_psm1 + trajectory_length_psm2
        print(f"\nPSM1 trajectory length: {trajectory_length_psm1:.2f} cm")
        print(f"PSM2 trajectory length: {trajectory_length_psm2:.2f} cm")
        print(f"Total trajectory length: {total_trajectory_length:.2f} cm")
        
        
        # Save the trajectory lengths
        psm1_trajectory_lengths.append(trajectory_length_psm1)
        psm2_trajectory_lengths.append(trajectory_length_psm2)
        total_trajectory_lengths.append(total_trajectory_length)
        
        # Adjust the plot 
        if demo_idx == 0:
            if display_legend_flag:
                # Add a legend
                ax.legend()
            
            if display_title_flag:
                # Set labels
                ax.set_xlabel('X Position (cm)', fontsize=12)
                ax.set_ylabel('Y Position (cm)', fontsize=12)
                ax.set_zlabel('Z Position (cm)', fontsize=12)
                title = f'PSM1 and PSM2 Positions (Tissue {selected_tissue} - Task {selected_task_idx})' if not recovery_flag else f'PSM1 and PSM2 Positions ({selected_tissue} - recovery)'
                ax.set_title(title, fontsize=14)

            elev = 30 # 10 # 30  # Adjust as needed
            azim = 20 # 45 # 90  # Adjust as needed

            # Set the viewing angle to be parallel to the Z plane
            ax.view_init(elev=elev, azim=azim)  # Adjust elev and azim as needed

    # Output the mean and std of the trajectory lengths
    mean_psm1_trajectory_length = np.mean(psm1_trajectory_lengths)
    mean_psm2_trajectory_length = np.mean(psm2_trajectory_lengths)
    mean_total_trajectory_length = np.mean(total_trajectory_lengths)
    std_psm1_trajectory_length = np.std(psm1_trajectory_lengths)
    std_psm2_trajectory_length = np.std(psm2_trajectory_lengths)
    std_total_trajectory_length = np.std(total_trajectory_lengths)
    print("\n-----------------------------------")
    print(f"\nMean PSM1 trajectory length: {mean_psm1_trajectory_length:.2f} cm +/- {std_psm1_trajectory_length:.2f} cm")
    print(f"Mean PSM2 trajectory length: {mean_psm2_trajectory_length:.2f} cm +/- {std_psm2_trajectory_length:.2f} cm")
    print(f"Mean total trajectory length: {mean_total_trajectory_length:.2f} cm +/- {std_total_trajectory_length:.2f} cm")


    # Save the plot
    plt.tight_layout()
    os.makedirs(output_dir_path, exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f'task_trajectory_overlay_plot_{dataset_name}_{selected_tissue}_{selected_task}_{selected_demo_dir}.png')
    plt.savefig(output_file_path)
    print(f"\nPlot saved at: {output_file_path}")
    
    # Save svg version
    svg_file_path = output_file_path.replace('.png', '.svg')
    plt.savefig(svg_file_path)
    print(f"SVG plot saved at: {svg_file_path}")

    # Show the plot
    plt.show()
    
    
if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')

    datasets_path = os.getenv("DATASET_PATH")
    dataset_name = "data_collector_1"
    selected_tissue = "tissue_31"
    selected_task_idx = 8
    recovery_flag = False
    display_title_flag = False
    display_legend_flag = False
    output_dir_path = os.path.join(os.getenv("OUTPUTS_PATH"), 'TrajectoryPlots', "task_overlay")

    task_dirs = os.listdir(os.path.join(datasets_path, dataset_name, selected_tissue))
    possible_tasks_for_task_idx = [task_dir for task_dir in task_dirs if task_dir.startswith(f"{selected_task_idx}_")]
    selected_task = possible_tasks_for_task_idx[0]
    selected_task = selected_task.replace("_recovery", "")
    recovery_exists_criterium = len([task for task in possible_tasks_for_task_idx if "recovery" in task]) != 0
    if recovery_flag and recovery_exists_criterium:
        selected_task = selected_task + "_recovery"
    task_dir_path = os.path.join(datasets_path, dataset_name, selected_tissue, selected_task)
    
    print(f"Visualizing task trajectory overlay for {selected_tissue} - Task {selected_task_idx} (recovery: {recovery_flag})")
    visualize_task_trajectory_overlay(task_dir_path, selected_tissue, selected_task_idx, display_legend_flag, display_title_flag, recovery_flag)