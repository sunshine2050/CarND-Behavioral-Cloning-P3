import csv
import cv2
import numpy as np

# list to append the csv lines
lines = []
#open the driving_log.csv & extract the data
with open('../data/driving_log.csv') as csvfile:
          reader = csv.reader(csvfile)
          for line in reader:
            lines.append(line)

# list to append images from ../data/IMG/*.jpg        
images = []
# list to append measurements from driving_log.csv        
measurements = []
# make the left and right camera images to act as 
for row in lines[1:]:
    steering_center = float(row[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.3 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = "../data/IMG/" 

    img_center = cv2.imread(path + row[0].split('/')[-1])
    img_left = cv2.imread(path + row[1].split('/')[-1])
    img_right = cv2.imread(path + row[2].split('/')[-1])
    
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)    
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)   

# add flipped images to make the netwark generalize better
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement)
    augmented_measurements.append(measurement * -1.0)
    
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)        

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# My Network model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,(5,5), strides=(2,2), activation = "relu"))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation = "relu"))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,(3,3), activation = "relu"))
model.add(Convolution2D(64,(3,3), activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the network
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
# Save weights to 'model.h5'
model.save('model.h5')        