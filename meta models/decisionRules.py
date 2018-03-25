from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import argparse
import sys
import pickle

def maxDecision():

    with open('/decision/image_decision_cv3_test.pkl','rb') as f:  
        x_test_image, y_test = pickle.load(f)
    with open('decision/image_decision_cv3_val.pkl','rb') as f: 
        x_val_image, y_val = pickle.load(f)

    print("loaded images")
    
    with open('decision/text_decision_cv3_test.pkl','rb') as f:  
        x_test_text = pickle.load(f)
    with open('/decision/text_decision_cv3_val.pkl','rb') as f:  
        x_val_text = pickle.load(f)

    x_test = []
    x_val = []
    for i in range(0,len(x_test_image)):
        joint = np.concatenate([x_test_image[i],x_test_text[i]])
        x_test = np.concatenate([x_test,joint])
         
    for i in range(0,len(x_val_image)):
        joint = np.concatenate([x_val_image[i],x_val_text[i]])
        x_val = np.concatenate([x_val,joint])

    x_val = x_val.reshape(len(x_val_image),12)
    x_test = x_test.reshape(len(x_test_image),12)
    accuracies = np.ones(12)#change accuracy to per-class estimates for weigthed rule
    
    print(accuracies)
    x_new = np.zeros((len(x_test),6))
    correct_predictions=0
    predictions = 0
    for i in range(0,len(x_test)):
      for j in range(0,12):
          x_test[i][j]*=accuracies[j]
      for j in range(0,6):
        x_new[i][j]= x_test[i][j] +x_test[i][j+6]

    predict = np.argmax(x_new,axis=1)
    for i in range (0,len(x_test)):
        if y_test[i][int(predict[i])]==1:
            correct_predictions+=1
        predictions+=1
    print(correct_predictions)
    accuracy = correct_predictions/predictions
    print("accuracy is "+str(accuracy))

