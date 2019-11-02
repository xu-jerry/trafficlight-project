import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import os
import yaml
from PIL import Image, ImageDraw
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

PATH = 'model'
net.load_state_dict(torch.load(PATH))

WIDTH = 1280
HEIGHT = 720


def get_all_labels(input_yaml, riib=False, clip=True):
    
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                         images[i]['path']))

        # There is (at least) one annotation where xmin > xmax
        for j, box in enumerate(images[i]['boxes']):
            if box['x_min'] > box['x_max']:
                images[i]['boxes'][j]['x_min'], images[i]['boxes'][j]['x_max'] = (
                    images[i]['boxes'][j]['x_max'], images[i]['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                images[i]['boxes'][j]['y_min'], images[i]['boxes'][j]['y_max'] = (
                    images[i]['boxes'][j]['y_max'], images[i]['boxes'][j]['y_min'])

        # There is (at least) one annotation where xmax > 1279
        if clip:
            for j, box in enumerate(images[i]['boxes']):
                images[i]['boxes'][j]['x_min'] = max(min(box['x_min'], WIDTH - 1), 0)
                images[i]['boxes'][j]['x_max'] = max(min(box['x_max'], WIDTH - 1), 0)
                images[i]['boxes'][j]['y_min'] = max(min(box['y_min'], HEIGHT - 1), 0)
                images[i]['boxes'][j]['y_max'] = max(min(box['y_max'], HEIGHT - 1), 0)

        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted. Since they are stored in a different
        # folder, the path also needs modifications.
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images

images = get_all_labels('C:/Users/chfjtu/Desktop/TrafficLightProject/train.yaml')

i = 0

for image in images:  
    image_path = image['path']  

    folder = image_path.split('\\')[-2]
    image_id = image_path.split('\\')[-1]
  
    target_file = 'C:\\Users\\chfjtu\\Desktop\\TrafficLightProject\\dataset_train_rgb\\rgb\\train\\' + folder + '\\' + image_id
    print("Target Image")
    print(target_file)
    
    raw_img = Image.open(target_file)
  
    for box in image['boxes']:
        
        y1 = int(box['y_min'])
        x1 = int(box['x_min'])
        y2 = int(box['y_max'])
        x2 = int(box['x_max'])
        
        cropped_img = raw_img.crop((x1, y1, x2, y2))
        
        
        trans = transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transformed_img = trans(cropped_img)
        transformed_img = transformed_img.unsqueeze(0)
        output = net(transformed_img)
        
        max_value, max_index = output[0].max(0)
    
    
        if max_index == 0:
            outline = 'green'
        elif max_index == 1:
            outline = 'red'
        else: 
            outline = 'yellow'
            
        draw = ImageDraw.Draw(raw_img)
        draw.rectangle(((x1, y1),(x2, y2)), fill = None, outline = outline, width = 2)
    
    #raw_img.show()
    i = i+1
    raw_img.save('C:\\Users\\chfjtu\\Desktop\\TrafficLightProject\\Video\\' + str(i) + '.png')
    

'''   
target_file = 'C:\\Users\\chfjtu\\Desktop\\Cropped\\Test\\Yellow\\219340.png'
print("Target Image")
print(target_file)
    
img = Image.open(target_file)
trans = transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
im = trans(img)
im = im.unsqueeze(0)
output = net(im)
        
print(output)
'''
