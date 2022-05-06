import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from torchvision import datasets, models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

dataPath = "Data/images"
#onlyfiles = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]

num_classes = 2 #Number of classes in dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model(model_name, num_classes): #Initialize Resnet

    if model_name == "resnet":

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes) # Update last layer to binary classification (Dog/cat)
        input_size = 224 #"Finally, notice that inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224)."
        
        #model = model.to(device)
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    return model, input_size



image_size = 64
batch_size = 128



def readData():
    dataset = ImageFolder(dataPath, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor()]))
    #print(dataset.samples)
    train, val, test = splitData(dataset,0.8,0.2)
    return train, val, test

def splitData(dataset,nrTrain,nrVal):
    len1 = int(nrTrain*len(dataset))
    len2 = int(len(dataset) - len1)
    train, val = torch.utils.data.random_split(dataset, [len1,len2])
    val, test = torch.utils.data.random_split(val, [int(0.5*len(val)),len(val) - int(0.5*len(val))])
    return train, val, test

def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    pass

def main():
    train, val, test = readData()

    model,input_size = initialize_model("resnet", 2)
    
    
    dataloaders = {label: torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for label, data in zip(['train', 'val', 'test'],[train, val, test])}

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model = model,
                dataloaders = dataloaders,
                criterion = criterion,
                optimizer = optimizer)

    print(model)




main()