import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image


def rectify_wrist_cam_image(image, angle, shift_x=0, shift_y=0):
    """Rectify the wrist camera image by rotating and shifting."""
    
    # If Image object then convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
        image_flag = True
    else:
        image_flag = False
    
    # Rotate the image
    rotated_image = rotate_image(image, angle)

    # Shift the image
    shifted_image = shift_image(rotated_image, shift_x, shift_y)

    # If previuosly image object then transf back
    if image_flag:
        return Image.fromarray(shifted_image)
    else:
        return shifted_image


# ----


def draw_center_line(image):
    """Draw a vertical center line on the image."""
    (h, w) = image.shape[:2]
    center_x = w // 2
    cv2.line(image, (center_x, 0), (center_x, h), (0, 255, 0), 2)
    return image


def process_images(image_path):
    """Process images from the given directory."""
    
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Rectify the image
    angle = -52.0
    shift_x, shift_y = 10, 0 
    rectified_image = rectify_wrist_cam_image(image, angle, shift_x, shift_y)

    # Draw the center line on the rectified image
    rectified_image_with_line = draw_center_line(rectified_image)
    
    # Visualize the original and rectified images
    visualize_rectified_image(original_image, rectified_image_with_line, image_path, angle)


def visualize_rectified_image(original, rectified, path, angle):
    """Visualize the original and rectified images."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Rectified (Rotated by {angle:.2f} degrees)')
    plt.imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.suptitle(f'Image: {os.path.basename(path)}')
    plt.show()


if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')
    
    # Set the dataset path and tissue IDs
    dataset_path = os.path.join(os.getenv("DATASET_PATH"), "data_collector_1")
    tissue_names = [f"tissue_{id}" for id in range(3, 10)] # Tissues with varying PSM1 wrist camera orientation

    # Define the output directory for the rectified images
    output_dir = os.path.join(os.getenv("OUTPUTS_PATH"), "RectifiedImagesEvaluation")
    os.makedirs(output_dir, exist_ok=True)

    for tissue_name in tissue_names:
        # Get the tissue directory
        tissue_path = os.path.join(dataset_path, tissue_name)
        
        # Randomly select a task and a demonstration
        tasks = os.listdir(tissue_path)
        selected_task = random.choice(tasks)
        demos = os.listdir(os.path.join(tissue_path, selected_task))
        selected_demo = random.choice(demos)
        demo_dir = os.path.join(tissue_path, selected_task, selected_demo)

        print(f"Selected Tissue: {tissue_name}, Selected Task: {selected_task}, Demonstration: {selected_demo}")

        # Get the image paths for the selected demonstration
        psm1_img_paths = os.path.join(demo_dir, "endo_psm1")
        # Select one example image from the PSM1 wrist camera dir
        random_image = random.choice(os.listdir(psm1_img_paths))
        image_path = os.path.join(psm1_img_paths, random_image)
        
        print(f"Processing image: {image_path}")
    
        process_images(image_path)

