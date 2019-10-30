import cv2
import dlib
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

import urllib.request

from sklearn import metrics
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm,tqdm_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import re
import gdown
import keras

from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

warnings.filterwarnings("ignore")
def model_to_string(model):
    import re
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sms = "\n".join(stringlist)
    sms = re.sub('_\d\d\d','', sms)
    sms = re.sub('_\d\d','', sms)
    sms = re.sub('_\d','', sms)  
    return sms

###Getting the csv data loaded

dataset_url = 'https://drive.google.com/uc?id=1xFiYsULlQWWmi2Ai0fHjtApniP5Pscuf'
dataset_path = './ferdata.csv'
gdown.download(dataset_url, dataset_path, True)

###Getting the Dlib Shape predictor!

dlibshape_url = 'https://drive.google.com/uc?id=17D3D89Gke6i5nKOvmsbPslrGg5rVgOwg'
dlibshape_path ='./shape_predictor_68_face_landmarks.dat'
gdown.download(dlibshape_url, dlibshape_path, True)

###Getting the Xpure loaded

pureX_url = 'https://drive.google.com/uc?id=1CglpXodenZVrkaZehLtfykfQv8dcnfO9'
pureX_path = './pureX.npy'
gdown.download(pureX_url, pureX_path,True)

###Getting the Xdata loaded

dataX_url = 'https://drive.google.com/uc?id=1sIJGxUM6rNBcWxucs6iynDepeKU1Q56p'
dataX_path = './dataX.npy'
gdown.download(dataX_url, dataX_path, True)


###Getting the Ydata loaded

dataY_url = 'https://drive.google.com/uc?id=1Rfr0OP-hZO_UZfuOyMNR2RjNRAro85zE'
dataY_path = './dataY.npy'
gdown.download(dataY_url, dataY_path, True)

model = Sequential()
model.add(Dense(4, input_shape=(3,),activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mean_squared_error'])
                
model = Sequential()
model.add(Dense(4,input_shape = (3,), activation = 'sigmoid'))
model.add(Dense(1, activation = 'linear'))
model.compile(loss='mean_squared_error',
optimizer = 'adam',
metrics = ['mean_squared_error'])

model.fit(x, y)

y = model.predict_classes(x)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

model_1_answer = Sequential()
model_1_answer.add(Dense(4, input_shape = (3,), activation = 'relu'))
model_1_answer.add(Dense(2, activation = 'softmax'))
model_1_answer.compile(loss='categorical_crossentropy',
                        optimizer = 'adam', 
                        metrics = ['accuracy'])
model_1 = model_1_answer

label_map = {"0":"ANGRY","1":"HAPPY","2":"SAD","3":"SURPRISE","4":"NEUTRAL"}

#Load the 68 face Landmark file
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
"""
Returns facial landmarks for the given input image path
"""
def get_landmarks(image):
  
  
  #:type image : cv2 object
  #:rtype landmarks : list of tuples where each tuple represents 
  #                  the x and y co-ordinates of facial keypoints
  
  #Bounding Box co-ordinates around the face(Training data is 48*48(cropped faces))
  rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]

  #Read Image using OpenCV
  #image = cv2.imread(image_path)
  #Detect the Faces within the image
  landmarks = [(p.x, p.y) for p in predictor(image, rects[0]).parts()]
  return image,landmarks

"""
Display image with its Facial Landmarks
"""
def image_landmarks(image,face_landmarks):
  """
  :type image_path : str
  :type face_landmarks : list of tuples where each tuple represents 
                     the x and y co-ordinates of facial keypoints
  :rtype : None
  """
  radius = -4
  circle_thickness = 1
  image_copy = image.copy()
  for (x, y) in face_landmarks:
    cv2.circle(image_copy, (x, y), circle_thickness, (255,0,0), radius)
    
  plt.imshow(image_copy, interpolation='nearest')
  plt.show()
  
"""
Computes euclidean distance between 68 Landmark Points for our features
e_dist is a list of features that will go into our model.
Each feature is a distance between two landmark points, and every pair of points
must have a feature.
"""
  
def landmarks_edist(face_landmarks):
    e_dist = []
    for i in range(len(face_landmarks)):
        for j in range(len(face_landmarks)):
            if i!= j:
                e_dist.append(distance.euclidean(face_landmarks[i],face_landmarks[j]))
    return e_dist
perceptron_answer = Sequential()
perceptron_answer.add(Dense(units = 1024, input_shape = (4556,),kernel_initializer='glorot_normal',activation = 'relu'))
perceptron_answer.add(Dense(units = 512,kernel_initializer='glorot_normal' , activation = 'relu'))
perceptron_answer.add(Dense(units = 5, activation = 'softmax'))
    
perceptron_answer.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.95),
              metrics=['accuracy'])
perceptron = perceptron_answer

dataX = np.load('./dataX.npy')
dataY = np.load('./dataY.npy')

num_labels = len(set(dataY))
onehotY = keras.utils.to_categorical(dataY,num_labels)

X_train, X_test, y_train, y_test = train_test_split(dataX, onehotY, test_size=0.1, random_state=42)

####Standardize the data####################
dnn_scaler = StandardScaler()
dnn_scaler.fit(X_train)
X_train = dnn_scaler.transform(X_train)
X_test = dnn_scaler.transform(X_test)

pickle.dump(dnn_scaler, open("dnn_scaler.p", "wb"))

epochs = 20
batch_size = 64

checkpoint = ModelCheckpoint('best_dnn_model.h5', 
                             verbose=1, monitor='val_loss',save_best_only=True, 
                             mode='auto')  
#training the model
dnn_history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoint],
          validation_data=(X_test, y_test),
          shuffle=True)
          
model = Sequential()
model.add(Dense(1024, activation='relu',kernel_initializer='glorot_normal', input_shape=(4556,)))
model.add(Dropout(0.2))
model.add(Dense(512,kernel_initializer='glorot_normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,kernel_initializer='glorot_normal', activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax'))

#Compliling the model with SGD optimixer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])
              
              


#Saves the Best Model Based on Val Loss

checkpoint = ModelCheckpoint('best_dnn_model.h5', 
                             verbose=1, monitor='val_acc',save_best_only=True, 
                             mode='auto')  




#training the model
dnn_history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoint],
          validation_data=(X_test, y_test),
          shuffle=True)
          
model.save('dnn_model.h5'

