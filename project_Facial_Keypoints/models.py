## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def  conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers=[]
    convlayer=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                        stride=stride,padding=padding,bias=False)
    layers.append(convlayer)
    if batch_norm==True:
        batchnormlayer=nn.BatchNorm2d(out_channels)
        layers.append(batchnormlayer)
       
    #using Sequential container
    return nn.Sequential(*layers)
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #[(W−K+2P)/S]+1
        self.conv1 = conv(1,  32, kernel_size = 5,batch_norm=True) ##224-5/1 +1 = 220/2 = 110
        self.conv2 = conv(32, 64, kernel_size = 5,batch_norm=True) #110-5/1 +1 = 106/2 = 53
        self.conv3 = conv(64, 128, kernel_size = 5,batch_norm=True)# 53-5/1 +1 = 49/2 = 24
        self.conv4 = conv(128,256, kernel_size = 5,batch_norm=True)# 24-5/1 +1 = 20/2 = 10
        self.conv5 = conv(256,512, kernel_size = 5,batch_norm=False)# 10-5/1 +1 = 6/2 = 3
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1=  nn.Linear(3*3*512,8000)
        self.fc2 = nn.Linear(8000,1000)
        self.fc3 = nn.Linear(1000,136)
        self.dropout=nn.Dropout(0.25)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv1(x))) 
        x= self.pool(F.relu(self.conv2(x))) 
        x= self.pool(F.relu(self.conv3(x))) 
        x= self.pool(F.relu(self.conv4(x))) 
        x= self.pool(F.relu(self.conv5(x))) 
        # prep for linear layer by flattening output from convolution layers
        x = x.view(x.size(0), -1)
        
        # process through fully connected layers
        x= self.dropout(F.relu(self.fc1(x)))
        x= self.dropout(F.relu(self.fc2(x)))
        x= self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
