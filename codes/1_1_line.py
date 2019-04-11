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


X = []
Y = []


path = "data_set/"
images = os.listdir(path)
random.seed(42)
random.shuffle(images)

def eval_class(data):
    length = data[0]
    width = data[1]     
    angle = data[2]
    colour = data[3]
    base = 0
    if length=='0':
        if width == '0':
            if colour == '0':
                class_id = int(angle)
            else:
                class_id = 12 + int(angle)
        else:
            if colour == '0':
                class_id = 24+int(angle)
            else:
                class_id = 36+ int(angle)
    else:
        if width == '0':
            if colour == '0':
                class_id = 48+int(angle)
            else:
                class_id = 60 + int(angle)
        else:
            if colour == '0':
                class_id = 72+int(angle)
            else:
                class_id = 84+ int(angle)
    return class_id

print("loading the data...........")
for image in images:
    img=cv2.imread(path+image)
    X.append(img)
    data=image[:-4].split('_')
    Y.append(eval_class(data))
print(len(X), "images loaded successfully.")

X = np.asarray(X)
X=X.astype('float32')/255.0
Y=np.asarray(Y)
Y=to_categorical(Y,96)
print(X.shape)

split = train_test_split(X,Y,test_size=0.4, random_state=42)
(X_train,X_test,Y_train,Y_test) = split

print("Data split \n train set :",len(X_train),"\n test set :", len(X_test))
input = Input(shape=(28,28,3))
x = Conv2D(32,(7,7),padding = 'same')(input)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
features = Flatten()(x)
x = Dense(1024)(features)
x = Activation("relu")(x)
output = Dense(96,activation = 'softmax')(x)
network = Model(input,output)
network.summary()

loss = categorical_crossentropy
network.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])

print("\n\nStarting the training")
stats = network.fit(X_train,Y_train, epochs = 2, validation_data = (X_test,Y_test), batch_size=16,verbose = 1,shuffle = True,callbacks=[PlotLossesKeras()])
print("\n\n Training completed successfully, saving the weights\n")
network.save_weights("weights_final.h5")

network.load_weights("weights_final.h5")

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
np.savetxt("cm_line_dataset.txt", cm , fmt = '%0.2f')