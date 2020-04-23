import argparse 
import torch 
import numpy as np
import json
from process_image import process_image
from torch import optim
from PIL import Image
from torchvision import models

# Take and process the command line inputs 
parser=argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='The device on which the calculations are made (cuda/cpu)')
parser.add_argument('--topk',type=int,default=5,help='Prints ouf the chose number of top-classes and their probabilities')
parser.add_argument('--json',type=str,default='cat',help='Selects a json-file of the user''s choice')
in_args=parser.parse_args()



# Load the model an its settings
checkpoint=torch.load('ImageClassifier/checkpoint.pth')


# Define the global variable 'device' 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load the pre-trained network and its settings

if checkpoint['model']=='resnet18':
    model= models.resnet18(pretrained=True)
    model.fc=checkpoint['classifier']
else:
    model= models.vgg16(pretrained=True)
    model.classifier=checkpoint['classifier']

model.to(device)
model.eval()

model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx=checkpoint['class_to_idx']



#Classify an unknown image
#Open the image
image = Image.open('ImageClassifier/rose.jpg')

image=process_image(image).to(device).float()
# Add a batch-dimension to the model 
image=torch.unsqueeze(image, 0)
#Calculate log-probabilities
log_ps=model(image)
#Transform log-probabilities into "classical" probabilities (numbers between 0 and 1)
ps=torch.exp(log_ps)
#Show which one of the 102 output units have top-5 probabilities (Note:First unit= index 0, second unit=index 2, etc.)
top_ps,top_class=ps.topk(k=in_args.topk,dim=1)
#Take the class-to-index items and build a dictionary where every class as a value is sorted to an index as a key 
idx_to_class={value:key for key, value in model.class_to_idx.items()}
# Convert the the top-5-classes and their corresponding probabilities into a numpy array
top_class=top_class.data.cpu().numpy()[0]
top_ps=top_ps.data.cpu().numpy()[0]
# Sort the Class number to each index of the top-5 classes 
top_labels=[idx_to_class[value] for value in top_class]


#load a json-file which sorts maps the flower name to its class number
if in_args.json=='cat':
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        # Name each label with the flower names from the json-file
        names= [cat_to_name[value] for value in top_labels]
    print(names)
else:
    print(top_labels)

print(top_ps)


     
    



    
    




