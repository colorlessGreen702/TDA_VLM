from torchvision.datasets import CIFAR100
import numpy as np
import os

template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

# template = ["a photo of a {}."]


class CIFAR100C(CIFAR100):
    dataset_dir = "CIFAR-100-C"

    def __init__(self, root, corruption_type, severity=5, transform=None, target_transform=None):
        super(CIFAR100C, self).__init__(root, train=False, transform=transform, 
                                       target_transform=target_transform, download=True)
        self.template = template
        self.classnames = self.classes
        if corruption_type != 'original':

            self.corruption_type = corruption_type
            self.severity = severity
            
            
            # Load corrupted data

            data_path = f"{root}/CIFAR-100-C/{corruption_type}.npy"
            labels_path = f"{root}/CIFAR-100-C/labels.npy"
            
            self.data = np.load(data_path)
            self.targets = np.load(labels_path)
            
            # Select data for the specified severity level
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]