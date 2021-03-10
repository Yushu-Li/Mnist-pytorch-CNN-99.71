import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import _LRScheduler
from torchviz import make_dot, make_dot_from_trace

best_acc=0
acc=0

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if epoch <2:
            warmup_scheduler.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),optimizer.param_groups[0]['lr']))


def test(args, model, device, test_loader):
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
    global acc
    acc=100. * correct / len(test_loader.dataset)
    global best_acc
    if acc>best_acc:
        best_acc=acc
    if acc>99.65:
        torch.save(model.state_dict(), "best_acc_" + str(best_acc) + ".pth")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


torch.manual_seed(args.seed)
model = CNN().to(device)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# optimizer = optim.Adam(model.parameters(),
#                        lr=0.003, betas=(0.9, 0.999),
#                        eps=1e-08, weight_decay=0,
#                        amsgrad=False)
#
# # Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

optimizer = torch.optim.RMSprop(model.parameters(),
                        lr=args.lr,alpha=0.9,eps=1e-08,weight_decay=0.0)
iter_per_epoch = len(train_loader)
warmup_scheduler =  WarmUpLR(optimizer, iter_per_epoch * 1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)
    if epoch >=2:
        scheduler.step(acc)

if (args.save_model):
    torch.save(model.state_dict(), "best_acc_"+str(best_acc)+".pth")








