import load_data
import torch 
import argparse 

from torch import nn, optim 
from torchvision import models 
from load_data import train_loader, valid_loader, train_data

parser=argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='The device on which the calculations are made (cuda/cpu)')
parser.add_argument('--arch',type=str,default='vgg16',help='Choose a pre-trained neuronal network on which a classifier \n is set on top (resnet/vgg)')
parser.add_argument('--hu1',type=int,default=8192,help='Choose the number of hidden units in the first layer')
parser.add_argument('--hu2',type=int,default=4096,help='Choose the number of hidden units in the second layer')
parser.add_argument('--hu3',type=int,default=2048,help='Choose the number of hidden units in the third layer')
parser.add_argument('--lr', type=int,default=0.0001,help='The Learning Rate for Gradient Descent')
parser.add_argument('--ep',type=int,default='30',help='The number of training epochs')
in_args=parser.parse_args()

print('Device:', in_args.device)
print('Architecture:', in_args.arch)
print('Learning Rate:', in_args.lr)
print('Epochs:',in_args.ep)
print('Hidden Unit 1:',in_args.hu1)
print('Hidden Unit 2:',in_args.hu2)
print('Hidden Unit 3:',in_args.hu3)

# Define load a model 
if in_args.arch =='resnet18':
    model = models.resnet18(pretrained=True)
else: model = models.vgg16(pretrained=True)


# Set the device at which the calculations are made.
# If the gpu device is available AND the user does not prefer the cpu of his computer it is chosen 
device=torch.device('cuda' if torch.cuda.is_available() and (in_args.device!='cpu') else 'cpu')


# Freeze Parameters
for param in model.parameters():
    param.requires_grad=False
# Set the Values for the hidden units
hidden_unit1=in_args.hu1
hidden_unit2=in_args.hu2
hidden_unit3=in_args.hu3
#The Classifier
if in_args.arch=='resnet18':
    model.fc=nn.Sequential(    nn.Linear(512,hidden_unit1),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_unit1,hidden_unit2),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_unit2,hidden_unit3),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_unit3,102),
                               nn.LogSoftmax(dim=1))
    
    model_p=model.fc.parameters()
    last_layer=model.fc
#vgg16
else:
    model.classifier=nn.Sequential(nn.Linear(25088,hidden_unit1),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_unit1,hidden_unit2),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(hidden_unit2,hidden_unit3),
                               nn.ReLU(),
                               nn.Linear(hidden_unit3,102),
                               nn.LogSoftmax(dim=1))
    model_p=model.classifier.parameters()
    last_layer=model.classifier
# Set Model to Device 
model.to(device)
# Test
print('Set Device: DONE!')
#Define Optimizer and Loss Criterion
criterion=nn.NLLLoss()
print('Criterion: Defined!')
#Which Parameters Which Learning Rate 
optimizer=optim.Adam(model_p,lr=in_args.lr)
print('Optimizer: Defined!')





#TRAINING SECTION!!!!

epochs=in_args.ep
train_losses, valid_losses=[],[]
 
for e in range (epochs):
    train_loss=0
    valid_loss=0
    accuracy=0
    
    
    for images, labels in train_loader:
        images, labels=images.to(device),labels.to(device)
        
        
        #
        optimizer.zero_grad()
        #
        
        log_ps=model(images)
        loss=criterion(log_ps,labels)
        loss.backward()
        optimizer.step()
        
        # 
        train_loss+=loss.item()
    # Validation     
    # Freeze Gradient Parameters
    with torch.no_grad():
        # Set Model to Evaluation
        model.eval()
        for images, labels in valid_loader:
            images, labels= images.to(device), labels.to(device)
            
            #Forward Pass for Testing
            log_ps=model(images)
            loss=criterion(log_ps,labels)
            valid_loss+=loss.item()
            ps=torch.exp(log_ps)
            top_ps,top_class=ps.topk(1,dim=1)
            equal=top_class==labels.view(*top_class.shape)
            accuracy+=torch.mean(equal.type(torch.FloatTensor))
        
    
    model.train()
    print('Summary: \n\n')
    train_losses.append(train_loss/len(train_loader))
    valid_losses.append(valid_loss/len(valid_loader))
    print('Train Loss: {}'.format(train_loss/len(train_loader)))
    print('Validation Loss: {}'.format(valid_loss/len(valid_loader)))
    print('Accuracy: {}'.format(accuracy/len(valid_loader)))
    
    
    # Save the parameters 
    checkpoint={ 'arch':in_args.arch,
                 'model':model, 
                 'classifier':last_layer,
                 'state_dict':model.state_dict(),
                 'class_to_idx':train_data.class_to_idx,
                 'optimizer':optimizer.state_dict(),
                 'epochs':epochs,
                  'learn_rate':in_args.lr}
    torch.save(checkpoint,'ImageClassifier/checkpoint.pth')
    