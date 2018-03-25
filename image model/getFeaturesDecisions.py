#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import yaml
import math
import pickle
# Parameters
# ==================================================


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "cnn-text-classification-tf/runs/1521835575/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/2.0)
    return e_x / e_x.sum()





def batchpredictDecision(x_raw):
    # Data Parameters

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # CHANGE THIS: Load data. Load your own data here

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    dataset_name = cfg["datasets"]["default"]
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                         categories=cfg["datasets"][dataset_name]["categories"],
                                                         shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                         random_state=cfg["datasets"][dataset_name]["random_state"])

    #x_raw, y_test = data_helpers.load_data_labels(datasets)
    
    print(len(x_raw))
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            probabilities = []
            for x_test_batch in batches:
                batch_predictions = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_predictions[1] = batch_predictions[1].reshape(len(batch_predictions[1]),6)
                for i in range(0,len(batch_predictions[1])):
                    probabilities = np.concatenate([probabilities,softmax(batch_predictions[1][i])])
                
    probabilities = probabilities.reshape(int(len(probabilities)/6) ,6)
    print(probabilities.shape)
    return probabilities





def batchpredictFeature(x_raw):
    # Data Parameters

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # CHANGE THIS: Load data. Load your own data here

    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    dataset_name = cfg["datasets"]["default"]
    datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
                                                         categories=cfg["datasets"][dataset_name]["categories"],
                                                         shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                                         random_state=cfg["datasets"][dataset_name]["random_state"])

    #x_raw, y_test = data_helpers.load_data_labels(datasets)
    
    print(len(x_raw))
    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("concat").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            probabilities = []
            for x_test_batch in batches:
                batch_predictions = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_predictions[0] = batch_predictions[0].reshape(len(batch_predictions[0]),768)
                #print(batch_predictions[0][0].shape)
                for i in range(0,len(batch_predictions[0])):
                    probabilities = np.concatenate([probabilities,batch_predictions[0][i]])
                
    probabilities = probabilities.reshape(int(len(probabilities)/768) ,768)
    print(probabilities)
    print(probabilities.shape)
    return probabilities

        
labels={0:"damaged_infrastructure",1:"damaged_nature",2:"fires",3:"flood",4:"human_damage",5:"non_damage"}

def batchload(filedir):
    # used to obtain features/predictions on text dataset for later training of meta-classifier
    step=1
    x = []
    y = []
    image_names = []
    captions = []
    for label,classe in labels.items():
        images = os.listdir(filedir +"/"+classe + '/images')
        for image in images:
            caption=" "
            try:
                with open(filedir+"/"+classe+ "/text/"+image[:-4]+".txt", 'r',encoding='utf-8') as myfile:
                    caption=myfile.read().replace('\n', ' ')
            except EnvironmentError:
                print("Text File not Found: "+filedir+"/"+classe+ "/text/"+image[:-4]+".txt" +" cap:"+caption)
            image_file= filedir+"/"+classe+"/images/"+image
            image_names.append(image_file)
            captions.append(caption)
            y_example= np.array([0 , 0 ,0,0,0,0])
            y_example[label]=1
            y = np.concatenate([y,y_example])
            step+=1
    features = batchpredictDecision(captions)# change this to get decision or features as output
    print("number of images: "+str(step))
    print(len(captions))
    return features

def getFeatures():
    #tf.flags.DEFINE_string("checkpoint_dir", "C:/Users/Hussein/Documents/Research/Text Classification/cnn-text-classification-tf/runs/1521835575/checkpoints", "Checkpoint directory from training run")
    x =  batchload("/multimodal_cv/CV4-val")
    with open('text_decision_cv4_val', 'wb') as f:
        pickle.dump(x,f)
    x =  batchload("/multimodal_cv/CV4-train")
    with open('text_decision_cv4_train', 'wb') as f:
        pickle.dump(x,f)
    x =  batchload("/multimodal_cv/CV4-test")
    with open('text_decision_cv4_test', 'wb') as f:
        pickle.dump(x,f)
getFeatures()
    
