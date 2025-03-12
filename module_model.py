import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os, sys
import numpy as np
# Device configuration


# Custom Dataset Class for Loading Grayscale Images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, np_file_list, label_list, transform=None, sel_transform=True):
        self.image_dir = image_dir
        self.transform = transform # * using
        self.np_filenames = np_file_list
        self.label_list = label_list
        self.sel_transform = sel_transform

    def __len__(self):
        return len(self.np_filenames)

    def __getitem__(self, idx):
        img_np = os.path.join(self.image_dir, self.np_filenames[idx])
        label_load = self.label_list[idx]

        image_load = Image.open(img_np).convert('L')
        # print('before_transform.shape')
        
        # image =  torch.Tensor(np.load(img_np)) 
        if self.sel_transform:
            image_load = self.transform(image_load)
            # image_load = image_load.squeeze()
        else:
            image_load = np.array(image_load)
            image_load = torch.Tensor(image_load).unsqueeze(0)
            # print('image_load.shape', image_load.shape)

        # print('after_transform.shape')
        # print('image_load.shape', image_load.shape)
        return image_load, label_load

class LargeImageCNN(nn.Module):
    def __init__(self, size_multiply=128):
        super(LargeImageCNN, self).__init__()
        self.size_multiply = size_multiply

        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Set in_channels=1 for grayscale images
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # ! for change dataset
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Set in_channels=1 for grayscale images
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32 , kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.fc1 = nn.Linear(128 * 32 * 32, 512)  # ! For 256 * 256 : Adjust based on resized image dimensions

        self.fc1 = nn.Linear(self.size_multiply, 512)  # * For 1000 * 1000 : Adjust based on resized image dimensions
        self.fc2 = nn.Linear(512, 2)  # Adjust for the number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print('x.shape', x.shape)
        # sys.exit()
        # x = x.view(-1, 128 * 32 * 32)  # ! For 256 * 256, Flatten
        x = x.view(-1, self.size_multiply)  # ! For 1000 * 1000, Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x