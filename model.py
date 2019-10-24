import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D ,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split



lines = []

#read the log file
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#print(lines)
lines.pop(0)
# Split the image dataset for validation
training_set, validation_set = train_test_split(lines, test_size=0.2)

images = []
steering_angles = []


#removes the label from the data


#extract the images and steering angles from the log file
check = 1
for line in lines:
    for i in range(3):
        source_path = line[i]
        source_path = source_path.replace(" ","") #removes space that from the csv file
        file_name = 'data/'+source_path
        angle = float(line[3])
        steering_angles.append(angle)
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.open(file_name)
        images.append(image)
        
        #save a copy of the center image in grayscale
        if check == 1 and i==0:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(img_gray,cmap="gray")
            plt.savefig('./examples/gray_image.png')
            
        if check == 1 and i==0:
            plt.imshow(image)
            plt.savefig('./examples/center_image.png')
            
        if check == 1 and i==1:
            plt.imshow(image)
            plt.savefig('./examples/left_image.png')
         
        if check == 1 and i==2:
            plt.imshow(image)
            plt.savefig('./examples/right_image.png')
            
    check = check+1

argumented_images, argumented_steering_angles = [],[]
image_taken = 1
for image,steering_angles in zip(images,steering_angles):
#     argumented_images.append(image)
#     argumented_steering_angles.append(steering_angles)
    argumented_images.append(cv2.flip(image,1))
    argumented_steering_angles.append(steering_angles*-1)
    
    if image_taken ==1:
        original_image = image
        plt.imshow(original_image)
        plt.savefig('./examples/original_image.png')
        flipped_image = cv2.flip(image,1)
        plt.figure()
        plt.imshow(flipped_image)
        plt.savefig('./examples/flipped_image.png')
        
    image_taken=image_taken+1
    
#get the training data
X_train = np.array(argumented_images)
y_train = np.array(argumented_steering_angles)

#buid ing the model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))


model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
fit_generator = model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=5)

plt.figure()
plot_x = np.arange(1, 6)
plt.plot(plot_x, fit_generator.history['loss'])
plt.plot(plot_x, fit_generator.history['val_loss'])
plt.ylabel('model mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training dataset', 'validation dataset'], loc='upper right')
plt.savefig('./examples/training_loss.png')

model.save('model2.h5')
exit()


    