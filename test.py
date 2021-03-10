import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
import numpy as np
import time

def test(args, model, device, test_loader):
    start=time.time()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc=100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('test_time:',time.time()-start)


class CNN(nn.Module):
    def __init__(self):
        # Super function. It inherits from nn.Module and we can access everythink in nn.Module
        super().__init__()
        # Creating CNN model
        """
          [[Conv2D->relu]*2 -> BatchNormalization -> MaxPool2D -> Dropout]*2 -> 
          [Conv2D->relu]*2 -> BatchNormalization -> Dropout -> 
          Flatten -> Dense -> BatchNormalization -> Dropout -> Out
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3136,256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm1d(256),

        )

        self.fc2 = nn.Linear(256,10)
    def forward(self,x):

        x = self.conv(x)
        x = torch.softmax(self.fc2((self.fc(x))),dim=1)
        return x


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-weights', type=str, required=True, help='adress to model file')
args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000, shuffle=False, **kwargs)


net = CNN().to(device)
net.load_state_dict(torch.load(args.weights))
net.eval()
test(args,net,device,test_loader)

# for n_iter, (image, label) in enumerate(test_loader):
#     #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
#
#     image = image.cuda()
#     label = label.cuda()
#
#
#     output = net(image)
#     _, pred = output.topk(5, 1, largest=True, sorted=True)
#
#     label = label.view(label.size(0), -1).expand_as(pred)
#     correct = pred.eq(label).float()
#
#     if correct[:, :1].sum() == 0:
#         img = torchvision.utils.make_grid(image.cpu()).numpy()
#         plt.imshow(np.transpose(img, (1, 2, 0)))
#         print('label'+str(label)+'pred'+str(pred))
#         plt.show()

