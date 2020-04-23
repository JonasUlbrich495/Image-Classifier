# Imports 
# TO DO: Cut what we do not need 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models 
from PIL import Image


# Paths for Data Loading 
data_dir = 'ImageClassifier/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

""""Define the transforms for the training, validation, and testing sets."""
#Training 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomGrayscale(p=0.1),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
#Testing
test_transforms=transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Validation 
valid_transforms=transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder Input: Path, Transform Procedure (Defined above)
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data=datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
# Test
print('Loading:DONE!')

# Define the dataloaders Input: Data Set, Batch Size, Shuffle? (Y/N) 
train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)

valid_loader=torch.utils.data.DataLoader(valid_data,batch_size=64)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=64)
#Load Jason 
print('Loader Defining : DONE!')

#Test 
print('Load Data: DONE!')