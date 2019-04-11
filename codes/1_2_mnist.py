import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import os
import cv2
# import pydot
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras


from keras.datasets import mnist
(X_train,Y_train),(X1_test,Y1_test)=mnist.load_data()

X_train  = X_train.reshape(60000,28,28,1)
from keras.utils import to_categorical
Y_train = np.asarray(Y_train)
Y_train = to_categorical(Y_train)
X_train = np.asarray(X_train)
X_train=X_train.astype('float32')/255.0





split = train_test_split(X_train,Y_train,test_size=0.4, random_state=42)
(X_train,X_test,Y_train,Y_test) = split

print("Data split \n train set :",len(X_train),"\n test set :", len(X_test))
input = Input(shape=(28,28,1))
x = Conv2D(32,(3,3),padding = 'same')(input)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(3,3),padding = 'same')(x)
x = Activation("relu")(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.5)(x)
features = Flatten()(x)
x = Dense(1024)(features)
x = Activation("relu")(x)
output = Dense(10,activation = 'softmax')(x)
network = Model(input,output)
network.summary()

loss = categorical_crossentropy
network.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])

print("\n\nStarting the training")
stats = network.fit(X_train,Y_train, epochs = 2, validation_data = (X_test,Y_test), batch_size=16,verbose = 1,shuffle = True,callbacks=[PlotLossesKeras()])
print("\n\n Training completed successfully, saving the weights\n")
network.save_weights("weights_mnist_2.h5")

network.load_weights("weights_mnist_2.h5")

print("\n Getting the predictions on the test set\n")
predictions = network.predict(X_test, batch_size = 16, verbose =1)

predictions = np.argmax(predictions, axis = 1)
predictions = predictions.astype('int')
predictions=predictions.reshape(predictions.shape[0],)
pred_GT=np.argmax(Y_test, axis =1)
pred_GT=pred_GT.astype('int')

print("\nF1-scores:")
fs=f1_score(pred_GT,predictions,average = 'weighted')
print(fs)

print("\nConfusion matrices:")
cm=confusion_matrix(pred_GT,predictions)
print(cm)
np.savetxt('CM_mnist_2.txt',cm,fmt='%.2f')
