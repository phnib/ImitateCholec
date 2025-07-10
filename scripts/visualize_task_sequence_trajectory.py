import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from ImitateCholec.utils.helper import get_valid_demo_start_end_indices


def generate_task_sequence_trajectory(start_task, end_task, datasets_path, dataset_name, selected_tissue, display_title_flag, kinematics_step_size):
    """
    Generates a 3D trajectory plot for a sequence of tasks, showing the PSM1 and PSM2 positions over time.
    
    Parameters:
    - start_task: The starting task index (inclusive).
    - end_task: The ending task index (inclusive).
    - datasets_path: The path to the datasets directory.
    - dataset_name: The name of the dataset to visualize.
    - selected_tissue: The tissue name to visualize.
    - display_title_flag: Whether to display the title on the plot.
    - kinematics_step_size: The step size for downsampling the kinematics data for plotting.
    """

    # Init the plot
    fig = plt.figure(figsize=(10, 8), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    # Init the total trajectory length variables
    procedure_trajectory_length_psm1 = 0
    procedure_trajectory_length_psm2 = 0
    procedure_total_trajectory_length = 0

    # Generate distinct colors for the number of tasks dynamically
    num_tasks = end_task - start_task + 1
    start_color_intensity = 0.3
    blue_colors = [cm.Blues(start_color_intensity + i / num_tasks * (1 - start_color_intensity)) for i in range(num_tasks)]
    red_colors = [cm.Reds(start_color_intensity + i / num_tasks * (1 - start_color_intensity)) for i in range(num_tasks)]

    for task_idx in range(start_task, end_task + 1):
        task_tasks = os.listdir(os.path.join(datasets_path, dataset_name, selected_tissue))
        selected_task = [task_task for task_task in task_tasks if task_task.startswith(f"{task_idx}_") and "recovery" not in task_task][0]
        task_task_path = os.path.join(datasets_path, dataset_name, selected_tissue, selected_task)
        demos_list = os.listdir(task_task_path)
        demo_idx = random.randint(0, len(demos_list) - 1) # Choose a random demo
        selected_demo_task = os.listdir(task_task_path)[demo_idx]
        demo_task_path = os.path.join(task_task_path, selected_demo_task)   
        kinematics_csv_path = os.path.join(demo_task_path, 'kinematics.csv')
        demo_kinematics = pd.read_csv(kinematics_csv_path)

        # ----------------------------------------

        # Get valid start and end positions 
        start, end, *_ = get_valid_demo_start_end_indices(demo_task_path)

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

        start_point_psm1, start_point_psm2 = psm1_position[:, 0], psm2_position[:, 0]
        if task_idx - start_task > 0:
            distance_between_tasks_psm1 = np.linalg.norm(start_point_psm1 - end_point_psm1)
            distance_between_tasks_psm2 = np.linalg.norm(start_point_psm2 - end_point_psm2)
            print(f"\nDistance between the start and end points of tasks {task_idx-1} and {task_idx}:")
            print(f"PSM2: {distance_between_tasks_psm2:.2f} cm, PSM1: {distance_between_tasks_psm1:.2f} cm")
        end_point_psm1, end_point_psm2 = psm1_position[:, -1], psm2_position[:, -1]

        # Assign colors and labels for the current task
        blue_color = blue_colors[task_idx-1]
        red_color = red_colors[task_idx-1]

        # Plot psm1 positions in blue
        start_point_size = 25
        other_points_size = 8
        ax.scatter(psm1_position[0, 0], psm1_position[1, 0], psm1_position[2, 0], c=[blue_color], label=f'psm1 (task {task_idx})', s=start_point_size)
        ax.scatter(psm1_position[0, 1::kinematics_step_size], psm1_position[1, 1::kinematics_step_size], psm1_position[2, 1::kinematics_step_size], c=[blue_color], s=other_points_size)
        ax.plot(psm1_position[0], psm1_position[1], psm1_position[2], c=blue_color, linewidth=0.5)

        # Plot psm2 positions in red
        ax.scatter(psm2_position[0, 0], psm2_position[1, 0], psm2_position[2, 0], c=[red_color], label=f'psm2 (task {task_idx})', s=start_point_size)
        ax.scatter(psm2_position[0, 1::kinematics_step_size], psm2_position[1, 1::kinematics_step_size], psm2_position[2, 1::kinematics_step_size], c=[red_color], s=other_points_size)
        ax.plot(psm2_position[0], psm2_position[1], psm2_position[2], c=red_color, linewidth=0.5)

        if task_idx == 1:
            # Set labels
            ax.set_xlabel('X Position (cm)', fontsize=12)
            ax.set_ylabel('Y Position (cm)', fontsize=12)
            ax.set_zlabel('Z Position (cm)', fontsize=12)
            if display_title_flag:
                ax.set_title('PSM1 and PSM2 Positions', fontsize=14)

            elev = 30   # Adjust as needed
            azim = 20   # Adjust as needed

            # # Set the viewing angle to be parallel to the Z plane
            ax.view_init(elev=elev, azim=azim)  # Adjust elev and azim as needed

        # Calculate avg and std distance between the points (wrt the kinematics_step_size)
        psm1_distances = np.linalg.norm(np.diff(psm1_position[:, ::kinematics_step_size], axis=1), axis=0)
        psm2_distances = np.linalg.norm(np.diff(psm2_position[:, ::kinematics_step_size], axis=1), axis=0)
        avg_psm1_distance = np.mean(psm1_distances)
        avg_psm2_distance = np.mean(psm2_distances)
        std_psm1_distance = np.std(psm1_distances)
        std_psm2_distance = np.std(psm2_distances)
        print(f"\nAvg distance between the points (wrt step size {kinematics_step_size} --> num points: {psm1_position[:, ::kinematics_step_size].size}):")
        print(f"PSM1: {avg_psm1_distance:.2f} cm +/- {std_psm1_distance:.2f} cm")
        print(f"PSM2: {avg_psm2_distance:.2f} cm +/- {std_psm2_distance:.2f} cm")

        # Calculate the trajectory lengths
        trajectory_length_psm2 = np.sum(np.linalg.norm(np.diff(psm2_position, axis=1), axis=0))
        trajectory_length_psm1 = np.sum(np.linalg.norm(np.diff(psm1_position, axis=1), axis=0))
        total_trajectory_length = trajectory_length_psm1 + trajectory_length_psm2
        print(f"\nPSM1 trajectory length (Task {task_idx}): {trajectory_length_psm1:.2f} cm")
        print(f"PSM2 trajectory length (Task {task_idx}): {trajectory_length_psm2:.2f} cm")
        print(f"Total trajectory length (Task {task_idx}): {total_trajectory_length:.2f} cm")
        procedure_trajectory_length_psm1 += trajectory_length_psm1
        procedure_trajectory_length_psm2 += trajectory_length_psm2
        procedure_total_trajectory_length += total_trajectory_length
        if task_idx != end_task:
            print("\n-----")

    # Total trajectory lengths
    print("\n----------------------------------------")
    print(f"\nTotal PSM1 trajectory length (tasks {start_task} to {end_task}): {procedure_trajectory_length_psm1:.2f} cm")
    print(f"Total PSM2 trajectory length (tasks {start_task} to {end_task}): {procedure_trajectory_length_psm2:.2f} cm")
    print(f"Total trajectory length (tasks {start_task} to {end_task}): {procedure_total_trajectory_length:.2f} cm")

    # Distance between the tasks start and end points


    if display_legend_flag:
        # Add a legend
        ax.legend()
    plt.tight_layout()

    # Save the trajectory lengths
    os.makedirs(output_task_path, exist_ok=True)
    output_file_path = os.path.join(output_task_path, f'task_sequence_trajectory_plot_{dataset_name}_{selected_tissue}_task_{start_task}_to_{end_task}.png')
    plt.savefig(output_file_path)
    print(f"Saved plot to {output_file_path}")
    
    # Save svg version
    svg_file_path = output_file_path.replace('.png', '.svg')
    plt.savefig(svg_file_path, dpi=300)
    print(f"Saved SVG plot to {svg_file_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env') 
    
    # Set parameters
    datasets_path = os.getenv("DATASET_PATH")
    dataset_name = "data_collector_1"
    selected_tissue = "tissue_31"
    start_task = 1
    end_task = 17
    plot_task_name = "full_procedure" if start_task == 1 and end_task == 17 else "multiple_tasks"
    kinematics_step_size = 7
    output_task_path = os.path.join(os.getenv("OUTPUTS_PATH"), 'TrajectoryPlots', plot_task_name)
    display_title_flag = False
    display_legend_flag = False
    
    #  Generate the task sequence trajectory
    generate_task_sequence_trajectory(start_task, end_task, datasets_path, dataset_name, selected_tissue, display_title_flag, kinematics_step_size)