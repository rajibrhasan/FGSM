import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from PIL import Image
from constant import STATS


class BaseDataset(Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.imdb = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, index):
        image, label = self.imdb[index]
        image = self.transforms(image)
        return image, label

def get_tranforms(dset_name, train):
    _transforms = []
    mean, std = STATS[dset_name]

    if dset_name.lower() == 'mnist':
        _transforms.append(transforms.Grayscale(num_output_channels=3))
    
    _transforms.append(transforms.Resize(224))
    if train:
        _transforms.append(transforms.RandomHorizontalFlip())
    
    _transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    return transforms.Compose(_transforms)
     
    
def create_dataset(dir, dset_name, train = True, download = True):
    assert dset_name in ["cifar10", "mnist"], f"Unsupported dataset: {dset_name}"

    _transforms = get_tranforms(dset_name, train)

    if dset_name.lower() == "cifar10":
        data = CIFAR10(dir, train = train, download = download)
    
    elif dset_name.lower() == "mnist":
        data = MNIST(dir, train = train, download = download)

    dataset = BaseDataset(data, _transforms)
    
    return dataset