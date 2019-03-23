# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/recover_2.jpg "Recovery Image"
[image4]: ./examples/recover_1.jpg "Recovery Image"
[image5]: ./examples/recover_3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flip.jpg "Flipped Image"
[image8]: ./examples/architectue.png "CNN Architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

(model.py lines 58-72) 
Model published by the autonomous vechile team at NVIDIA 
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers added two dropout layers to overcome overfitting
![alt text][image8]


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 64, 69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect data, import the data then use an architecture used to train a real car this architecture is more powerfull than LeNET which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers (model.py lines 64, 69)
 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track I noticed that car does't behave correctly in the sand area so I retrained it using the training mode in the simulator for this particular case & I added the data to train the model with

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-72)  consisted of a convolution neural network with the following layers and layer sizes 

![alt text][image8]
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the given data. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the network more unbiased to a direction 
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had about 25200 images including images from right & left cameras after flipping images and appending them, then mapping right & left images to the center image I had around 40000 images I then preprocessed this data by normalizing & cropping the unneeded pixels I used 10 epochs & the learning rate was set using Adam's optimizer.


I finally put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
