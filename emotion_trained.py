
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# preparing training data

# test and train data current directory
train_dir=(r"...\train") 
test_dir=(r"...\test")
emotions=["angry","happy","sad","surprise"]


def emotion_data(directory,limit):
    x=[]
    y=[]
    for emotion in emotions:
        path=os.path.join(directory,emotion)
        label=emotions.index(emotion)
        inc=0
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            if img_array is None:
                continue
            inc=inc+1
            x.append(img_array)
            y.append(label)
            if inc==limit:
                break
    return np.array(x,dtype='float32') , np.array(y)


x_train,y_train=emotion_data(train_dir,1000)
x_test,y_test=emotion_data(test_dir,100)

x_train=x_train/255
x_test=x_test/255


import matplotlib.pyplot as plt
def plot_img(index):
    plt.figure(figsize=(1,1))
    plt.imshow(x_train[index],cmap='gray')
    plt.xlabel(emotions[y_train[index]])


# making the deep learning model
classes=len(emotions)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model=Sequential([
    data_augmentation,
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(classes,activation='softmax')
])
model.compile(optimizer = 'adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs=30)


model.evaluate(x_test,y_test)
# After training
model.save('emotion_model.h5')







