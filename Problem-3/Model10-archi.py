# Convolutional Neural Network
# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution and max pooling layers
classifier.add(Convolution2D(64, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dense(output_dim = 2, activation = 'softmax'))

# Compiling the CNN
sgd = optimizers.SGD(lr=0.01)
classifier.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# trainset 0.8
training_set = train_datagen.flow_from_directory('/dataset/trainset',
                                                 target_size = (256, 256),
                                                 batch_size = 8,
                                                 classes=['dogs', 'cats'],
                                                 class_mode = 'binary')
# validaset 0.2
validation_set = test_datagen.flow_from_directory('/dataset/validaset',
                                            target_size = (256, 256),
                                            batch_size = 2,
                                            classes=['dogs', 'cats'],
                                            class_mode = 'binary')
history=classifier.fit_generator(training_set,
                         samples_per_epoch = 16000,
                         nb_epoch = 25,
                         validation_data = validation_set,
                         nb_val_samples = 3998)

classifier.save("model-10")
with open('history-14-02-model-10.txt','w') as f: f.write(str(history.history))
print(history.history)
