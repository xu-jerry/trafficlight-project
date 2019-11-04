import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PATH = 'model'

net.load_state_dict(torch.load(PATH))

def accuracy (loader):
    correct = 0
    total = 0
    i = 0
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i = i + 1
            if i % 200 == 0:
                print(i)
    print('Accuracy of the network on the ' +str(len(traindataset)) + ' images of the current dataset: %d %%' % (
    100 * correct / total))
    print('Correct:' + str(correct))
    print('Total:' + str(total))

print("Train dataset:")
accuracy(trainloader)
print("Validation dataset:")
accuracy(validationloader)
print("Test dataset:")
accuracy(testloader)

j = 0
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))

with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels)
        j = j + 1
        if j % 200 == 0:
            print(j)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(3):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100.0 * class_correct[i] / class_total[i]))
print(len(traindataset))
print(class_correct)
print(class_total)
