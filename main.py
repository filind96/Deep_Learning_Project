import torch as to
import numpy as np
from torchvision import datasets, transforms

dataPath = "/Data/images"



def main():
    print("Hello World")


main()


def readData():
    images = datasets.ImageFolder(dataPath, transform=transform)
    

    return images