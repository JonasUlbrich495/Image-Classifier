'''Prepare an image, that it can used in the predict function later on'''

from PIL import Image
import numpy as np
import torch

def process_image(img):
    ratio=img.size[0]/img.size[1]
    #Resize the image 
    if ratio<=1:
        width,height=256,int(256*(1/ratio))
    else:width,height=int(256*ratio),256
    img=img.resize((width,height))
    
    #Crop the image to 224,224
    new_width=224
    new_height=224

    left=(img.size[0]-new_width)/2
    top=(img.size[1]-new_height)/2
    right=(img.size[0]+new_width)/2
    bottom=(img.size[1]+new_height)/2
    
    img=img.crop((left,top,right,bottom))
    
    # Normalize and transpose the image for further processes 
    img=np.array(img)
    img=img/255
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img=(img-mean)/std
    img=img.transpose(2,0,1)
    
    img=torch.tensor(img)
    
   
    return img
        