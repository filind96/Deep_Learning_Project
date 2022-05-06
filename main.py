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
onlyfiles = ["Data/images/Abyssinian_1.jpg"]
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
image_size = 64
batch_size = 128



def readData():
    #images = datasets.ImageFolder(dataPath, transform=transform)
    dataset = ImageFolder(dataPath, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor()]))
    print(dataset(0))
    # preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # for file in onlyfiles:
    #     image = Image.open(file)
    #     input_tensor = preprocess(image)
    #     input_batch = input_tensor.unsqueeze(0)

    #     if torch.cuda.is_available():
    #         input_batch = input_batch.to('cuda')
    #         model.to('cuda')
        
    #     with torch.no_grad():
    #         output = model(input_batch)

    #     print(output[0])

    #     probabilities = nn.functional.softmax(output[0], dim=0)
    #     print(probabilities)
    #     print(image)
    

    return dataset

def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))


def main():
    dataset = readData()



main()