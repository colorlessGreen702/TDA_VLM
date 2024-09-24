from torchvision.datasets import CIFAR10
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

classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR10C(CIFAR10):
    dataset_dir = "CIFAR-10-C"

    def __init__(self, root, corruption_type, severity=5, transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(root, train=False, transform=transform, 
                                       target_transform=target_transform, download=True)
        self.template = template
        self.classnames = classnames
        if corruption_type != 'original':

            self.corruption_type = corruption_type
            self.severity = severity
            
            
            # Load corrupted data

            data_path = f"{root}/CIFAR-10-C/{corruption_type}.npy"
            labels_path = f"{root}/CIFAR-10-C/labels.npy"
            
            self.data = np.load(data_path)
            self.targets = np.load(labels_path)
            
            # Select data for the specified severity level
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]