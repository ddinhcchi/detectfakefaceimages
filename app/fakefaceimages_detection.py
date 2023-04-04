import torch
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
# import numpy as np
# import os
# import pandas as pd 
# import cv2
from torch.nn import functional as F
from torchsummary import summary
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])

def load_image(image_path):
    with open(image_path,'rb') as f:
        image = Image.open(f).convert('L')
        image = transform(image)
    return image

class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()
        self.conv2 = nn.Conv2d(1, 128,kernel_size = 3,padding = 1)
        self.conv3 = nn.Conv2d(128, 256,kernel_size = 3,padding = 1)
        self.conv4 = nn.Conv2d(256, 512,kernel_size = 3,padding = 1)
        self.conv5 = nn.Conv2d(512, 512,kernel_size = 3,padding = 1)
        self.conv6 = nn.Conv2d(512, 512,kernel_size = 4,padding = 0)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2) 
        self.linear1 = nn.Linear(512*1*1,2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        x= self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x= self.dropout(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv5(x)))        
        x = F.relu(self.conv6(x))
        x = x.view(-1,512*1*1)
        x = self.linear1(x)  
        return x

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath, map_location=device)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model
    
def testing(path_image):
    path_model = './model/checkpoint.pth'
    model = Detect()
    model = load_checkpoint(path_model, model)
    image = load_image(path_image)
    perdiction = model(image).argmax(dim = -1).item()
    if(perdiction == 0):
        return "FAKE"
    else: 
        return "REAL"

# path = "../test_images/01.jpg"
# print(testing(path))

