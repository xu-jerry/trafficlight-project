# Traffic Light Project

Using the Bosch dataset, this is a machine learning project that will be able to identify whether a trafficlight in an image is red, green, or, yellow. The Bosch dataset has test data from 24068 to 40734, evens, resulting in 8334 images.

## Demo Video
[![IMAGE ALT TEXT HERE](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/TrafficLightTestVideo.png)](https://www.youtube.com/watch?v=G4B4tAR6vx4)

## Data Visualization
### Green light
<img src="https://raw.githubusercontent.com/xu-jerry/trafficlight-project/master/Images/green_labeled.png" width="300">


Challenging case image, and data distribution

Many of the images were either occluded, off, or too small. For some, even a human eye would not be able to detect which color the light was.

## Convolutional Neural Network

Convolutional neural networks (CNN) are used in machine learning, specifially when dealing with images, in order to build models that can label data. They are conprised of layers of "neurons" that are very similar to brain neurons- they take information from the previous layer and processes it, giving an output of the modified information. In Convolutional Neural Networks, they give the input tensor a weight and bias using the formula y = wx + b.

The input for this CNN is an image and the output is a matrix of probabilities, size three. The first number is green, the second is red, and the third is yellow. The greater the number, the more likely the model thinks it is.

This is a graph of what my CNN looks like:

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/Model%20Architecture.png" width = "300">

## Training Process

To train, I used the decay learning rate

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
