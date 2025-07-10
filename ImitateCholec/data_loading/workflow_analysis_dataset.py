import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from ImitateCholec.utils.helper import get_all_sorted_task_dirs


class CholecWorkflowAnalysisDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        split_name,
        all_splits_data_collector_tissue_dict,
        input_transforms=None,
        stride=1,
    ):
        super().__init__()
        
        # Set attributes
        self.dataset_dir = dataset_dir
        self.split_name = split_name
        self.all_splits_data_collector_tissue_dict = all_splits_data_collector_tissue_dict
        self.input_transforms = input_transforms
        self.stride = stride 

        # Get all images and labels (surgical task + recovery flag)  
        self.image_paths_list, self.task_label_indices_list, self.recovery_flag_list = [], [], []
        self.task_labels = get_all_sorted_task_dirs()
        for data_collector, split_tissue_dict in self.all_splits_data_collector_tissue_dict.items():
            tissue_list = split_tissue_dict[self.split_name]
            for tissue in tissue_list:
                tissue_dir_path = os.path.join(self.dataset_dir, data_collector, tissue)
                task_dirs = os.listdir(tissue_dir_path)
                task_dirs = sorted(task_dirs, key=lambda p: int(p.split('_')[0]))  # Sort by task number
                for task_dir in task_dirs:
                    
                    # Skip task 1 (grasping gallbladder) as the task demonstrations from tissue 1 to 8 have a different definition as of tissue 9 onwards
                    if task_dir.startswith('1_'): 
                        continue 
                    
                    # Load all demo directories for the current task
                    task_dir_path = os.path.join(tissue_dir_path, task_dir)
                    demo_dirs = os.listdir(task_dir_path)
                    for demo_dir in demo_dirs:
                        demo_dir_path = os.path.join(task_dir_path, demo_dir)
                        left_img_dir_path = os.path.join(demo_dir_path, "da_vinci_stereo_left") # NOTE: Here only using the left img dir (as normally done for surgical workflow analysis) 
                        
                        # Get all image paths for the current demo
                        image_paths = sorted(
                            [os.path.join(left_img_dir_path, f) for f in os.listdir(left_img_dir_path) if f.endswith('.jpg')],
                            key=lambda x: int(os.path.basename(x).split('.')[0].replace('frame', ''))
                        )
                        
                        # Add images and labels to the lists
                        self.image_paths_list.extend(image_paths[::self.stride])
                        label_idx = self.task_labels.index(task_dir.replace("_recovery", ""))  # Get the index of the task in the classes list
                        self.task_label_indices_list.extend([label_idx] * len(image_paths[::self.stride]))
                        if 'recovery' in demo_dir:
                            self.recovery_flag_list.extend([1] * len(image_paths[::self.stride]))
                        else:
                            self.recovery_flag_list.extend([0] * len(image_paths[::self.stride]))
    
    
    def __len__(self):
        return len(self.image_paths_list)


    def __getitem__(self, idx):
        # Get the image and label index
        image_path = self.image_paths_list[idx]
        label_idx = self.task_label_indices_list[idx]
        recovery_flag = self.recovery_flag_list[idx]
        
        # Load raw image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.input_transforms is not None:
            image = self.input_transforms(image)
        
        return image, label_idx, recovery_flag
        
    
if __name__ == "__main__":
    # Load the ImitateCholec.env file
    from dotenv import load_dotenv
    load_dotenv('ImitateCholec.env')   

    from ImitateCholec.utils.transforms import get_transforms, destandardize

    # Set all parameters
    data_collector_tissue_dict = {
        "data_collector_1": {
            "train": [
                "tissue_1", "tissue_2", "tissue_3", "tissue_4", "tissue_6", "tissue_7",
                "tissue_8", "tissue_9", "tissue_10", "tissue_11", "tissue_12", "tissue_13",
                "tissue_15", "tissue_16", "tissue_17", "tissue_18", "tissue_19", "tissue_20",
                "tissue_21", "tissue_22", "tissue_23", "tissue_28", "tissue_29", "tissue_30",
                "tissue_31", "tissue_32", "tissue_33", "tissue_34"
            ],
            "val": [
                "tissue_5", "tissue_14", "tissue_27"
            ],
            "test": []
        },
        "data_collector_2": {
            "train": [],
            "val": [],
            "test": ["tissue_24", "tissue_25", "tissue_26"] # Test on the different data collector (to check for robustness)
        }
    }
    split_name = "val"
    image_dim=(224, 224)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet mean and std
    apply_augmentations_on_train = False
    input_transforms = get_transforms(split_name, image_size=(224, 224), mean=mean, std=std,
                                      apply_augmentations_on_train=apply_augmentations_on_train)
    dataset_dir = os.getenv("DATASET_PATH")
    stride = 10

    # Create the dataset
    chole_dataset = CholecWorkflowAnalysisDataset(
        dataset_dir=dataset_dir,
        split_name=split_name,
        all_splits_data_collector_tissue_dict=data_collector_tissue_dict,
        input_transforms=input_transforms,
        stride=stride
    )

    # Save a random image
    idx = np.random.randint(0, len(chole_dataset))
    img, label_idx, recovery_flag = chole_dataset[idx]
    label = chole_dataset.task_labels[label_idx]
    recovery_label = "True" if recovery_flag == 1 else "False"
    img_destandardized = destandardize(img, mean, std)
    plt.imshow(img_destandardized.permute(1, 2, 0))
    plt.title(f"Label: {label}, Recovery: {recovery_label}", pad=5, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    # --
    outputs_dir = os.path.join(os.getenv("OUTPUTS_PATH"), "DatasetExamples")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_path = os.path.join(outputs_dir, "workflow_analysis_dataset_example.jpg")
    plt.savefig(output_path)
    print("Example image saved as 'example_dataset_image.jpg'")
