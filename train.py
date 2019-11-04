import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('Green', 'Red', 'Yellow')

# functions to show an image
def load_dataset(data_path):
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, 
                                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, 
                                               shuffle=True, num_workers=0)
    return train_dataset, train_loader
    

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataset, _ = load_dataset('C:/Users/chfjtu/Desktop/Cropped/Train/')
traindataset, validationdataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=4, 
                                               shuffle=True, num_workers=0)
validationloader = torch.utils.data.DataLoader(validationdataset, batch_size=4, 
                                               shuffle=True, num_workers=0)
testdataset, testloader = load_dataset('C:/Users/chfjtu/Desktop/Cropped/Test/')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
prevloss = 0
lr = 0.001

for epoch in range(10):  # loop over the dataset multiple times

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('[%d] training loss: %.5f' %
        (epoch + 1, running_loss / len(trainloader)))
    currloss = running_loss / len(trainloader)
    if abs(prevloss - currloss) < lr * 5:
        lr = lr / 5
    prevloss = running_loss / len(trainloader)
    
    validation_running_loss = 0.0
    for i, data in enumerate(validationloader, 0):
        inputs, labels = data

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        validation_running_loss += loss.item()
    print('[%d] validation loss: %.5f' %
        (epoch + 1, validation_running_loss / len(validationloader)))

print('Finished Training')

PATH = 'model'

torch.save(net.state_dict(), PATH)

