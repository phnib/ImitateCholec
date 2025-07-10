import os

import numpy as np
import cv2
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ImitateCholec.utils.helper import get_valid_demo_start_end_indices


def quaternion_to_rotation_matrix(x, y, z, w):
    """
    Convert a quaternion into a rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.identity(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Compute rotation matrix elements
    R = np.array([
        [1 - 2 * y * y - 2 * z * z,     2 * x * y - 2 * z * w,       2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,         1 - 2 * x * x - 2 * z * z,   2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,         2 * y * z + 2 * x * w,       1 - 2 * x * x - 2 * y * y]
    ])
    return R

def draw_orientation(ax, position, rotation_matrix, length=1.5):
    """
    Draw the orientation of a tool using its rotation matrix.

    Args:
    - ax: Matplotlib 3D axes object
    - position: np.array of shape (3,) indicating the position of the tool
    - rotation_matrix: np.array of shape (3,3) indicating the rotation matrix of the tool
    - length: Length of the orientation axes
    - color: Color of the orientation axes
    """
    origin = position
    x_axis = rotation_matrix[:, 0] * length
    y_axis = rotation_matrix[:, 1] * length
    z_axis = rotation_matrix[:, 2] * length

    ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color='b', arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.1)

def visualize_pose_reconstruction(psm1_history, psm2_history, fig_3d_plot, ax_3d_plot,
                                elev=45, azim=225, x_range=(-100, 100), y_range=(-100, 100), z_range=(-100, 100),
                                coord_frame_length=1, coord_frame_vis_flag=True, print_jaw_values_flag=True):
    """
    Visualize the 3D reconstruction of the tool tip positions and orientations for both PSM1 and PSM2,
    showing the last n points and their orientations.

    Inputs:
    - psm1_history: deque containing tuples of (position, rotation_matrix, jaw value (optionally)) for PSM1
    - psm2_history: deque containing tuples of (position, rotation_matrix, jaw value (optionally)) for PSM2
    - fig_3d_plot: Matplotlib figure object for plotting
    - ax_3d_plot: Matplotlib axes object for plotting
    - elev: The elevation angle of the plot
    - azim: The azimuth angle of the plot
    - x_range: The x-axis range of the plot
    - y_range: The y-axis range of the plot
    - z_range: The z-axis range of the plot
    - coord_frame_length: Length of the rotation axes
    - coord_frame_vis_flag: Flag to visualize the coordinate frame
    - print_jaw_values_flag: Flag to display jaw values in the title

    Returns:
    - image_3d_position: The image representation of the 3D plot in BGR format suitable for OpenCV
    """

    # Clear last plot
    ax_3d_plot.clear()

    psm1_jaw_values = None
    psm2_jaw_values = None

    # Plot PSM1 History
    if psm1_history:
        psm1_positions = np.array([pose[0] for pose in psm1_history])
        psm1_rotations = [pose[1] for pose in psm1_history]
        ax_3d_plot.plot(psm1_positions[:,0], psm1_positions[:,1], psm1_positions[:,2], c="b", label='PSM1 Trajectory')
        if coord_frame_vis_flag:
            for pos, rot in zip(psm1_positions, psm1_rotations):
                draw_orientation(ax_3d_plot, pos, rot, length=coord_frame_length)
        else:
            # Scatter plot for all except the last point
            ax_3d_plot.scatter(psm1_positions[:-1,0], psm1_positions[:-1,1], psm1_positions[:-1,2], c="b", marker='o', s=3)
        # Highlight the last point
        latest_psm1_pos = psm1_positions[-1]
        ax_3d_plot.scatter(latest_psm1_pos[0], latest_psm1_pos[1], latest_psm1_pos[2], c="b", marker='o', s=15)
        ax_3d_plot.text(latest_psm1_pos[0], latest_psm1_pos[1], latest_psm1_pos[2], 'PSM1', color='b')
        
        # Extract jaw values if available
        if print_jaw_values_flag and len(psm1_history[0]) > 2:
            psm1_jaw_values = np.round(np.array([pose[-1] for pose in psm1_history]), 1)

    # Plot PSM2 History
    if psm2_history:
        psm2_positions = np.array([pose[0] for pose in psm2_history])
        psm2_rotations = [pose[1] for pose in psm2_history]
        ax_3d_plot.plot(psm2_positions[:,0], psm2_positions[:,1], psm2_positions[:,2], c="r", label='PSM2 Trajectory')
        if coord_frame_vis_flag:
            for pos, rot in zip(psm2_positions, psm2_rotations):
                draw_orientation(ax_3d_plot, pos, rot, length=coord_frame_length)
        else:
            # Scatter plot for all except the last point
            ax_3d_plot.scatter(psm2_positions[:-1,0], psm2_positions[:-1,1], psm2_positions[:-1,2], c="r", marker='o', s=3)
        # Highlight the last point
        latest_psm2_pos = psm2_positions[-1]
        ax_3d_plot.scatter(latest_psm2_pos[0], latest_psm2_pos[1], latest_psm2_pos[2], c="r", marker='o', s=15)
        ax_3d_plot.text(latest_psm2_pos[0], latest_psm2_pos[1], latest_psm2_pos[2], 'PSM2', color='r')
        
        # Extract jaw values if available
        if print_jaw_values_flag and len(psm2_history[0]) > 2:
            psm2_jaw_values = np.round(np.array([pose[-1] for pose in psm2_history]), 1)

    # Set plot limits and labels
    ax_3d_plot.set_xlim(x_range)
    ax_3d_plot.set_ylim(y_range)
    ax_3d_plot.set_zlim(z_range)
    ax_3d_plot.set_xlabel('X Axis (m)')
    ax_3d_plot.set_ylabel('Y Axis (m)')
    ax_3d_plot.set_zlabel('Z Axis (m)')
    
    # Set the title - include jaw values if available and flag is enabled
    if print_jaw_values_flag and (psm1_jaw_values is not None or psm2_jaw_values is not None):
        title = '3D Reconstruction of Tool Poses with Trajectory History\n'
        if psm1_jaw_values is not None:
            title += f'PSM1 Jaw: {psm1_jaw_values[-1]:.1f}'
        if psm2_jaw_values is not None:
            if psm1_jaw_values is not None:
                title += ', '
            title += f'PSM2 Jaw: {psm2_jaw_values[-1]:.1f}'
        ax_3d_plot.set_title(title)
    else:
        ax_3d_plot.set_title('3D Reconstruction of Tool Poses with Trajectory History')
    
    ax_3d_plot.legend()

    # Set a consistent view angle
    ax_3d_plot.view_init(elev=elev, azim=azim)

    # Draw the plot
    fig_3d_plot.canvas.draw()

    # Convert the Matplotlib figure to an OpenCV image
    image_3d_position = np.frombuffer(fig_3d_plot.canvas.buffer_rgba(), dtype=np.uint8).reshape(-1, 4)[:, :3]
    height, width = fig_3d_plot.canvas.get_width_height()[::-1]
    image_3d_position = image_3d_position.reshape(height, width, 3)
    image_3d_position = cv2.cvtColor(image_3d_position, cv2.COLOR_RGB2BGR)

    return image_3d_position


def generate_video_vision_kinematics_single_demo(base_path, selected_tissue, selected_task, selected_demo,
                                               output_dir_path, kinematics_vis_step_size=10, history_length=10,
                                               coord_frame_vis_flag=True, x_limits=[-5, 2], y_limits=[-2.5, 2.5], z_limits=[4, 8],
                                               print_jaw_values_flag=True, output_video_suffix=""):
    print(f"Creating video for tissue: {selected_tissue}, task: {selected_task}, demo: {selected_demo}")
    
    # Variables for scaling and translating the kinematics for better visualization
    unit_factor = 100  # E.g., 100 for cm, 1000 for mm
    x_offset, y_offset, z_offset = 0, 2, 0

    # Paths to the demo
    demo_dir_path = os.path.join(base_path, selected_tissue, selected_task, selected_demo)
    kinematics_csv_path = os.path.join(demo_dir_path, 'kinematics.csv')

    # Load kinematics data
    demo_kinematics = pd.read_csv(kinematics_csv_path)

    # Get valid start and end positions
    start, end, _ = get_valid_demo_start_end_indices(demo_dir_path)

    # Define video writer
    final_video_path = os.path.join(output_dir_path, f"{selected_tissue}_{selected_task}_{selected_demo}_single_demo_vision_kinematics{output_video_suffix}.mp4")
    out = None

    # Initialize history buffers with a maximum length of 'history_length'
    psm1_history = deque(maxlen=history_length)
    psm2_history = deque(maxlen=history_length)

    # Create a Matplotlib figure and axes for 3D plotting
    fig_3d_plot = plt.figure(figsize=(6, 6))
    ax_3d_plot = fig_3d_plot.add_subplot(111, projection='3d')

    # List all image files
    images_dir_path = os.path.join(demo_dir_path, "da_vinci_stereo_left")
    images_sorted = sorted([image for image in os.listdir(images_dir_path) if image.endswith(".jpg")])

    # Process images for each frame index
    for frame_counter, image_name in enumerate(images_sorted[start:end+1]):
        if frame_counter % kinematics_vis_step_size == 0:
            kinematics_row = demo_kinematics.iloc[start + frame_counter]

            # Extract positions and orientations for PSM2
            psm2_position = np.array([
                kinematics_row['psm2_pose.position.x'] * unit_factor - x_offset,
                kinematics_row['psm2_pose.position.y'] * unit_factor - y_offset,
                kinematics_row['psm2_pose.position.z'] * unit_factor - z_offset
            ], dtype=np.float32)
            psm2_orientation = [
                kinematics_row['psm2_pose.orientation.x'],
                kinematics_row['psm2_pose.orientation.y'],
                kinematics_row['psm2_pose.orientation.z'],
                kinematics_row['psm2_pose.orientation.w']
            ]
            psm2_rotation_matrix = quaternion_to_rotation_matrix(*psm2_orientation)
            psm2_jaw = kinematics_row['psm2_jaw']

            # Extract positions and orientations for PSM1
            psm1_position = np.array([
                kinematics_row['psm1_pose.position.x'] * unit_factor + x_offset,
                kinematics_row['psm1_pose.position.y'] * unit_factor + y_offset,
                kinematics_row['psm1_pose.position.z'] * unit_factor + z_offset
            ], dtype=np.float32)
            psm1_orientation = [
                kinematics_row['psm1_pose.orientation.x'],
                kinematics_row['psm1_pose.orientation.y'],
                kinematics_row['psm1_pose.orientation.z'],
                kinematics_row['psm1_pose.orientation.w']
            ]
            psm1_rotation_matrix = quaternion_to_rotation_matrix(*psm1_orientation)
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
        
        image_path = os.path.join(images_dir_path, image_name)
        image = cv2.imread(image_path)

        # Resize visualization image to match the height of the other images
        vis_height, vis_width = visualization_image.shape[:2]
        target_height = image.shape[0]
        scaling_factor = target_height / vis_height
        new_vis_width = int(vis_width * scaling_factor)
        visualization_image_resized = cv2.resize(visualization_image, (new_vis_width, target_height))

        # Concatenate the images (original image and visualization)
        final_image = cv2.hconcat([image, visualization_image_resized])

        # Write the final image to the video
        if out is None:
            h, w, _ = final_image.shape
            out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        out.write(final_image)

    # Release the video writer
    if out:
        out.release()
    print(f"Video saved at {final_video_path}")


def plot_psm_position_trajectory(kinematics_chunk, unit_factor=100, x_offset=0, y_offset=2, z_offset=0, 
                                 kinematics_step_size=1, display_title_flag=True, ax=None, fig=None,
                                 x_limits=[-5, 2], y_limits=[-2.5, 2.5], z_limits=[4, 8]):
    """
    Plot PSM1 and PSM2 position trajectory from kinematics chunk.
    
    Args:
        kinematics_chunk: pandas DataFrame with kinematics data (positions only)
        unit_factor: Unit conversion factor (100 for cm, 1000 for mm)
        x_offset, y_offset, z_offset: Offset values for positioning
        kinematics_step_size: Step size for plotting trajectory points
        display_title_flag: Whether to display plot title
        ax: Matplotlib 3D axes object (if None, creates new)
        fig: Matplotlib figure object (if None, creates new)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for the trajectory
    blue_color = 'blue'
    red_color = 'red'
    
    # Extract PSM positions
    psm1_position = np.array([
        kinematics_chunk['psm1_pose.position.x'] * unit_factor + x_offset,
        kinematics_chunk['psm1_pose.position.y'] * unit_factor + y_offset,
        kinematics_chunk['psm1_pose.position.z'] * unit_factor + z_offset
    ], dtype=np.float32)
    
    psm2_position = np.array([
        kinematics_chunk['psm2_pose.position.x'] * unit_factor - x_offset,
        kinematics_chunk['psm2_pose.position.y'] * unit_factor - y_offset,
        kinematics_chunk['psm2_pose.position.z'] * unit_factor - z_offset
    ], dtype=np.float32)
    
    # Plot psm1 positions in blue
    start_point_size = 25
    other_points_size = 8
    if len(psm1_position[0]) > 0:
        ax.scatter(psm1_position[0, 0], psm1_position[1, 0], psm1_position[2, 0], 
                  c=[blue_color], label=f'PSM1', s=start_point_size)
        if len(psm1_position[0]) > 1:
            ax.scatter(psm1_position[0, 1::kinematics_step_size], psm1_position[1, 1::kinematics_step_size], 
                      psm1_position[2, 1::kinematics_step_size], c=[blue_color], s=other_points_size)
        ax.plot(psm1_position[0], psm1_position[1], psm1_position[2], c=blue_color, linewidth=1.5)
    
    # Plot psm2 positions in red
    if len(psm2_position[0]) > 0:
        ax.scatter(psm2_position[0, 0], psm2_position[1, 0], psm2_position[2, 0], 
                  c=[red_color], label=f'PSM2', s=start_point_size)
        if len(psm2_position[0]) > 1:
            ax.scatter(psm2_position[0, 1::kinematics_step_size], psm2_position[1, 1::kinematics_step_size], 
                      psm2_position[2, 1::kinematics_step_size], c=[red_color], s=other_points_size)
        ax.plot(psm2_position[0], psm2_position[1], psm2_position[2], c=red_color, linewidth=1.5)
    
    # Set labels
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)
    ax.set_xlabel('X Position (cm)', fontsize=12)
    ax.set_ylabel('Y Position (cm)', fontsize=12)
    ax.set_zlabel('Z Position (cm)', fontsize=12)
    
    if display_title_flag:
        ax.set_title('PSM1 and PSM2 Position Trajectory', fontsize=14)
    
    # Set viewing angle
    elev = 30
    azim = 20
    ax.view_init(elev=elev, azim=azim)
    
    ax.legend()
    
    return fig, ax