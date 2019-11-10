# Traffic Light Project

Using the [Bosch dataset](https://hci.iwr.uni-heidelberg.de/node/6132), this is a machine learning project that will be able to identify whether a traffic light in an image is red, green, or, yellow. The Bosch dataset has testing and training data on their website, but I decided to switch them because the training data had a couple of errors in the annotations. In addition, I wanted the training to have more images than test. My training dataset had 8334 images and my testing dataset had 5093 images, all with a resolution of 1280 x 720 pixels.

## Table of Contents

- [Demo Video](#demo-video)
- [Data Visulization](#data-visualization)
- [Convolutional Neural Network](#convolutional-neural-network)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Further Discussion](#further-discussion)
- [Contact](#contact)

## Demo Video
[![IMAGE ALT TEXT HERE](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/TrafficLightTestVideo.png)](https://www.youtube.com/watch?v=G4B4tAR6vx4)

This video shows the end result of my Convolutional Neural Network's classification of the annotated traffic lights. Because it is not 100% accurate, the colors might be off in some frames.

## Data Visualization

Examples of training images, with the traffic lights outlined in the color they are annotated in.

### Green light
<img src="https://raw.githubusercontent.com/xu-jerry/trafficlight-project/master/Images/green_labeled.png" width="300">

### Red Light
<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/red_labeled.png" width = "300">

### Yellow Light
<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/yellow_green_labeled.png" width = "300">

### Challenging Cases 
Many of the images were either occluded, off, or too small. For some, even a human eye would not be able to detect which color the light was.

### Occluded

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/occluded.png" width = "300">

### Small

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/small.png" width = "300">

### Data Distribution

#### Train
![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/train_distribution.png)  ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/train_width.png) ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/train_heights.png)  ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/train_size.png)

#### Test
![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/test_distribution.png)  ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/test_width.png) ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/test_height.png)  ![test](https://github.com/xu-jerry/trafficlight-project/blob/master/Images/test_size.png)

Here is the comparision of train versus test data. There are a lot fewer yellow data than green data, which contributes to the lower accuracy for data with ground truth yellow.

## Convolutional Neural Network

Convolutional neural networks (CNN) are used in machine learning, specifially when dealing with images, in order to build models that can label data. They are comprised of layers of "neurons" that are very similar to brain neurons- they take information from the previous layer and processes it, giving an output of the modified information. In convolutional neural networks, they give the input tensor a weight and bias using the formula y = wx + b. This is also known as a convolution layer. Other layers can have activation functions, which help the neuron decide if it will fire or not. Many activation functions are used today, but the most popular one and the one I used is what is known as ReLU, aka Rectified Linear Unit. Pooling layers reduce the number of parameters in the network and make the information easier to process. The last couple of layers are usually fully connected layers, where they are connected to all the activations from previous layers.

The input for this CNN is an image and the output is a matrix of probabilities, size three. The first number is green, the second is red, and the third is yellow. The greater the number, the more likely the model thinks it is.

This is a graph of what my CNN looks like:

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/Model%20Architecture.png" width = "600">

## Training Process

To train, I used a decaying learning rate, starting from 0.001 and decaying by 1/5 every time the change in loss is less than 5 times the learning rate. Trial and error taught me that 0.001 was the optimal learning rate to begin with because 0.01 would overshoot and 0.0001 would be too slow to converge. 

In order to see if my model was overfitting the data, I split the training data into train and validation, in an 8:2 ratio. Here is the plot of training loss versus validation loss:

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/TrainingValidationLoss.png" width = "300">

As you can see, the validation loss is always below the training loss, suggesting that there is no overfitting. Note that the y-axis is log scale.

## Evaluation

After training the model, I evaluated it on the dataset. Accuracy for the training data was 99% (4897/4898), validation was 100% (1225/1225), and test was 91% (2705/2941). Because it is a lot higher than 33%, I know that it was not randomly guessing and was actually able to identify most of the cases. 

Inside the test data, the model predicted 97% (1682/1730) of green, 92% (964/1042) of red, and 34% (59/169) of yellow. As expected, yellow was the most inaccurate, due to the very few training data of yellow.

### Confusion Matrix

<img src="https://github.com/xu-jerry/trafficlight-project/blob/master/Images/ConfusionMatrix.png" width = "300">

## Further Discussion
This entire project used annotations from the Bosch dataset. Later, this can expand to image segmentation so that it can identify where the traffic lights are from any image, in addition to classifying them. Also, this was a simplified version of the problem, with all the arrow cases removed and all the traffic lights smaller than 5 pixels wide and 10 pixels long removed. If this can expand further, we can implement this piece of code into a physical device than can be attached to a windshield, identifying traffic lights in real time.

## Contact
If you have any questions, feel free to contact me at xuchujun2672@gmail.com!
