import numpy as np
import os
import cv2 as cv
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta, sgd

image_size = (3,640,480)
cap = cv.VideoCapture('output_train.avi')


def load_train():
    X_train = []
    y_train = []
    speeds = pd.read_fwf('train.txt',header=None)
    y_train = np.array(speeds)
    total_frame = speeds.shape[0]
    curr_frame=0
    while(cap.isOpened()):
        curr_frame = curr_frame+1
        print('Current Frame :: ',curr_frame)
        if curr_frame>total_frame/50:
            print('All frames parsed.. Exiting...')
            break
        _,frame = cap.read()
        # cv.imshow('a',frame)
        # cv.waitKey(10)
        X_train.append(frame)
    X_train=np.array(X_train)
    cap.release()
    return X_train,y_train


def read_and_normalize_train_data():
    train_data, train_target = load_train()
    # train_data = (train_data/255).astype(np.float32)
    # print('Train shape:', train_data.shape)
    print(train_data.shape, 'train samples')
    print(train_target.shape, 'target shape')
    return train_data, train_target


def create_model():
    nb_filters = 8
    nb_conv = 5
    model = Sequential()

    # add a convolutional model first
    model.add(Convolution2D(nb_filters,nb_conv,border_mode='valid',input_shape=(480,640,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear')) #regression model, hence linear activation

    model.compile(loss='mean_squared_error', optimizer=sgd())
    model.summary()
    return model

def train_model(batch_size=60,nb_epoch=50):
    num_samples = 20400
    cv_size = 500

    train_data, train_target = read_and_normalize_train_data()
    train_data.reshape([-1,640,480,3])
    train_target = train_target[:train_data.shape[0],:]
    print(train_target.shape)

    # X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=cv_size, random_state=56741)
    model = create_model()

    model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model

train_model(nb_epoch=5)


