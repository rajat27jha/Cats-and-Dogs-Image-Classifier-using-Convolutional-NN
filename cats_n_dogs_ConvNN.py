import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
# loading data from pickle files
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0  # normalizing our data

model = Sequential()
#  creating an instance of Sequential() class

# adding a convolutional layer
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))  # using activation function as rectified linear
model.add(MaxPooling2D(pool_size=(2, 2)))  # creating a maxpooling layer

# same as above
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))  # adding a hidden/Dense layer

model.add(Dense(1))  # this is the output layer giving only one output as 0 or 1
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)