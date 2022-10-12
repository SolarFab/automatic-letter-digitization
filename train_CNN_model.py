
"""
This script serves to train the final CNN model with pictures of the companie's logos.
Training was done one a google colab server to accelerate the training of the CNN model:
- for the prototype, logos of 6 companies were used
- the mobilnet_v2 CNN model was used and trained with around 100 pictures of each company
- three layers were added to the mobilnet_V2 CNN model
- logos were saved in google drive
"""

import numpy as np
import pandas as pd
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import glob
from tensorflow.keras import preprocessing 
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image




# Import the data/fotos from the folder
# Each company has own folder with its fotos of the logo
# Folders are named after the company (e.g. vodafone)
images = []
labels = []
folders = glob.glob ("./images/*")
for folder in folders: 
  files = glob.glob(f'{folder}/*.png')
  for file in files:
      image = cv2.imread(file)
      images.append(image)
      labels.append(folder.split('/')[2])


# Prepare for Training the dataset
data_gen = preprocessing.image.ImageDataGenerator(
    # define the preprocessing function that should be applied to all images
    preprocessing_function=mobilenet_v2.preprocess_input,
    # fill_mode='nearest',
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True, 
    zoom_range=0.2,
    # shear_range=0.2 
    validation_split = 0.2   
)


#base_path = "/content/drive/MyDrive/brand_logos/"
base_path = "./images/"
classes = list(np.unique(labels))

# define training batch
train_batches = data_gen.flow_from_directory(directory=base_path,class_mode="categorical", \
target_size=(224,224), classes=classes, batch_size=64,subset='training')

#define validation batch
validation_batches = data_gen.flow_from_directory(directory=base_path,class_mode="categorical", \
target_size=(224,224), classes=classes, batch_size=64,subset='validation')

# a generator that returns batches of X and y arrays
train_data_gen = data_gen.flow_from_directory(
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=150,
        target_size=(224, 224)
)

# load in all images at once
xtrain, ytrain = next(train_data_gen)
xtrain.shape, ytrain.shape


### Create Model
### Select the convolutional base 
base_model = mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    alpha=0.35,         # specific parameter of this model, small alpha reduces the number of overall weights
    pooling='avg',      # applies global average pooling to the output of the last conv layer (like a flattening)
    include_top=False,  # !!!!! we only want to have the base, not the final dense layers 
    input_shape=(224, 224, 3)
)

### Freeze the weights
base_model.trainable = False

### Add your own dense layers on top
len(classes)
model = keras.Sequential()
model.add(base_model)
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(classes), activation='softmax')) #!!! Final layer with a length of classes, and softmax activation 

## Compile and fit the model"""
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

# observe the validation loss and stop when it does not improve after 3 iterations
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(xtrain, ytrain, 
          epochs=50, 
          verbose=2,
          batch_size=100, 
          callbacks=[callback],
          # use 30% of the data for validation
          validation_split=0.3)

### save the model 
model.save("./model/final_project_logo_webcam_fotos.h5")

