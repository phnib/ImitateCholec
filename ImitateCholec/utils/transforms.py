import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
import torch


def get_transforms(
    split='train', 
    image_size=(224, 224),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    apply_augmentations_on_train=True
):
    """
    Get transforms for different splits using RandAugment
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Target image size tuple (height, width)
        mean: Normalization mean values (default: ImageNet)
        std: Normalization std values (default: ImageNet)
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if split == 'train' and apply_augmentations_on_train:
        return transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            RandAugment(),  # Uses default num_ops=2, magnitude=9
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val/test
        return transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        
def destandardize(tensor, mean, std):
    """Destandardize a tensor using mean and std"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean