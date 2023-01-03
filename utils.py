import torch
import torchvision
import torchvision.transforms as transforms


def getMnistDataSubset(BS = 2000):

    transform = transforms.Compose(
        [transforms.ToTensor(),torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
     
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=False)
    
    return trainloader, trainset
    