
import numpy as np
from PIL import Image
import math
import statistics as stat
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
import json
import seaborn as sns
from preprocessing import *
from housekeeping import *
import matplotlib.pyplot as plt
import time




def baseline_model(model='baseline'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(11,)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        'accuracy',
        tf.keras.metrics.Precision(), 
        tf.keras.metrics.Recall()])

    return model

def larger_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(11,)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        'accuracy',
        tf.keras.metrics.Precision(), 
        tf.keras.metrics.Recall()])

    return model


def smaller_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=16, activation='relu', input_shape=(11,)))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        'accuracy',
        tf.keras.metrics.Precision(), 
        tf.keras.metrics.Recall()])

    return model


def train_test(X, y, verbose=False, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
        
    models = [baseline_model, smaller_model, larger_model]
    results = []
    for i, f in enumerate(models):
        start = time.time()
        model = f()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, verbose=0)
        y_pred = model.predict(X_test)
        y_pred = [1 if x > .5 else 0 for x in y_pred]

        res = metrics(y_pred, y_test)
        results.append(res)

        if i==2 and plot:
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('larger model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('larger model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    
        

if __name__ == "__main__":
    X, y = data_prep(lda=True, norm=True)
    train_test(X, y, plot=True)
