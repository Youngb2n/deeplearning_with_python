import cv2
import os

from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean = mean, std = std)


train_transform =transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

crop_size = 224

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    normalize,
])



def load_list(list_path, image_root_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_root_path, line[0]))
            labels.append(int(line[1]))
    return images, labels


class imagenet_dataset(Dataset):
    def __init__(self, image_paths = "", label_list = None, target_size = None, transform = None):
        
        self.image_list = image_paths
        self.label_list = label_list
        self.target_size = target_size
        self.transform = transform
    
    def __len__(self):
        if self.label_list:
            return len(self.label_list)

    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
def data_loader(train_path, valid_path, train_list_file, valid_list_file, train_batch, valid_batch, image_size=(224, 224, 3), train_shuffle=True, valid_shuffle=False):
    
    train_dir, train_list_dir = load_list(train_list_file, train_path)
    valid_dir, valid_list_dir = load_list(valid_list_file, valid_path)
    
    train_dataset = imagenet_dataset(train_dir, train_list_dir, target_size=image_size, transform=train_transform)
    valid_dataset = imagenet_dataset(valid_dir, valid_list_dir, target_size=image_size, transform=valid_transform)
    
    train_loader= DataLoader(train_dataset, batch_size=train_batch, shuffle=train_shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=valid_shuffle)
    
    return train_loader, valid_loader
