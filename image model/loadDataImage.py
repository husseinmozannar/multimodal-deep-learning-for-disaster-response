from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from predicttext import predictText,textRepresentation
import argparse
import sys
import pickle
from sklearn.preprocessing import normalize

graph="C:/tmp/frozen_inception_v3_cv1.pb"
labels={0:"damaged_infrastructure",1:"damaged_nature",2:"fires",3:"flood",4:"human_damage",5:"non_damage"}

input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer='input'  
output_layer=  'InceptionV3/Logits/Dropout_1b/Identity' #InceptionV3/Predictions/Reshape_1
decision_layer = "InceptionV3/Predictions/Reshape"
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/1.5)
    return e_x / e_x.sum()
  
def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph


def imageRepresentationF(file_name,Graph):
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = Graph.get_operation_by_name(input_name);
  output_operation = Graph.get_operation_by_name(output_name);
  config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))

  with tf.Session(graph=Graph ,config=config) as sess:
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
  results = np.reshape(results,(2048))

  return results

def imageRepresentationD(file_name,Graph):
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + decision_layer
  input_operation = Graph.get_operation_by_name(input_name);
  output_operation = Graph.get_operation_by_name(output_name);
  config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))

  with tf.Session(graph=Graph ,config=config) as sess:
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
  results = np.reshape(results,(7))#or 2048
  probs = softmax(results[1:])
  print(probs)
  return probs # entry 0 is a background class


def jointRepresentationF(image,caption,Graph,checkpoint):
    
  imageRepres = imageRepresentationF(image,Graph)
  #textRepres = textRepresentation(caption,checkpoint)
  #jointRepres = np.concatenate((imageRepres,textRepres),axis=0)
  return imageRepres
  

def jointRepresentationD(image,caption,Graph,checkpoint):
    
  imageRepres = imageRepresentationD(image,Graph)
  #textRepres = textRepresentation(caption,checkpoint)
  #jointRepres = np.concatenate((imageRepres,textRepres),axis=0)
  return imageRepres

def loadDataDecision(filedir):
    # this is just to start tensorflow once with restricted memory
    w = tf.Variable(tf.random_uniform([2, 2]))
    init_op = w.initializer
    config=tf.ConfigProto(
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))

    with tf.Session(config=config) as sess:
      sess.run(init_op)
    #end 
    print (graph)
    print (checkpoint)
    Graph = load_graph(graph)
    step=1
    x = []
    y = []
    for label,classe in labels.items():
        images = os.listdir(filedir +"/"+classe + '/images')
        for image in images:
            print (image)
            caption=" "
            try:
                with open(filedir+"/"+classe+ "/text/"+image[:-4]+".txt", 'r',encoding='utf-8') as myfile:
                    caption=myfile.read().replace('\n', ' ')
            except EnvironmentError:
                print("Text File not Found: "+filedir+"/"+classe+ "/text/"+image[:-4]+".txt" +" cap:"+caption)
            representation = jointRepresentationD(filedir+"/"+classe+"/images/"+image,caption,Graph,checkpoint)
            y_example= np.array([0,0,0,0,0,0])
            y_example[label]=1
            x= np.concatenate([x,representation])
            y = np.concatenate([y,y_example])
            print("step: "+str(step))
            step+=1
    x = x.reshape(step-1,6)
    y= y.reshape(step-1,6)
    return x,y


def loadDataFeatures(filedir):
    # this is just to start tensorflow once with restricted memory
    w = tf.Variable(tf.random_uniform([2, 2]))
    init_op = w.initializer
    config=tf.ConfigProto(
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))

    with tf.Session(config=config) as sess:
      sess.run(init_op)
    #end 
    print (graph)
    print (checkpoint)
    Graph = load_graph(graph)
    step=1
    x = []
    y = []
    for label,classe in labels.items():
        images = os.listdir(filedir +"/"+classe + '/images')
        for image in images:
            print (image)
            caption=" "
            try:
                with open(filedir+"/"+classe+ "/text/"+image[:-4]+".txt", 'r',encoding='utf-8') as myfile:
                    caption=myfile.read().replace('\n', ' ')
            except EnvironmentError:
                print("Text File not Found: "+filedir+"/"+classe+ "/text/"+image[:-4]+".txt" +" cap:"+caption)
            representation = jointRepresentationF(filedir+"/"+classe+"/images/"+image,caption,Graph,checkpoint)
            y_example= np.array([0,0,0,0,0,0])
            y_example[label]=1
            x= np.concatenate([x,representation])
            y = np.concatenate([y,y_example])
            print("step: "+str(step))
            step+=1

    x = x.reshape(step-1,2048)
    y= y.reshape(step-1,6)
    return x,y
          
                     
def loadImageData():
  # this is to get features/predictions of image dataset and save them
  global graph
  global checkpoint


  graph="C:/tmp/frozen_inception_v3_cv1.pb"
  checkpoint="C:/Users/Hussein/Documents/Research/Text Classification/cnn-text-classification-tf/runs/1521833797/checkpoints"
 
  x,y = loadDataDecision("C:/Users/Hussein/Documents/Research/Data/multimodal_cv/CV1-train")
  with open('image_decision_cv1_train.pkl', 'wb') as f:
    pickle.dump([x,y],f)
  x,y = loadDataDecision("C:/Users/Hussein/Documents/Research/Data/multimodal_cv/CV1-test")
  with open('image_decision_cv1_test.pkl', 'wb') as f:
    pickle.dump([x,y],f)
  x,y = loadDataDecision("C:/Users/Hussein/Documents/Research/Data/multimodal_cv/CV1-val")
  with open('image_decision_cv1_val.pkl', 'wb') as f:
    pickle.dump([x,y],f)



