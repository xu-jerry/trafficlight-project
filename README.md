# Traffic Light Project

Using the Bosch dataset, this is a machine learning project that will be able to identify whether a trafficlight in an image is red, green, or, yellow. The Bosch dataset has test data from 24068 to 40734, evens, resulting in 8334 images.

## Video Demo
[![IMAGE ALT TEXT HERE](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/TrafficLightTestVideo.png)](https://www.youtube.com/watch?v=a2UpQiP4zbA)

## Data Visualization
### Green light
<img src="https://raw.githubusercontent.com/xu-jerry/trafficlight-project/master/Images/green_labeled.png" width="300">


Challenging case image, and data distribution

Many of the images were either occluded, off, or too small. For some, even a human eye would not be able to detect which color the light was.

## Convolutional Neural Network

Describe how CNN works and draw your model architecture. Briefly discuss why you choose the model.

## Training Process

decay learning rate

linear classifier
large learning rates
other hyperparameters

plot - training loss, validation loss

## Evaluation

Train, validation, and test accuracy
report the confusion matrix

## Further Discussion
Annotation issue
Simplied version (remove arrows, remove occluded case, remove tiny crops)
we need detector/tracker
