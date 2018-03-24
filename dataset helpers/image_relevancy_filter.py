from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import argparse
import sys

directory = "" # of image and text data in format : category/images and category/text for each category

graph="" # of relevancy filter classifier
output_layer='final_result:0' # of graph
input_layer='DecodeJpeg/contents:0'#of graph


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

  return predictions

def predictImage(file_name,graph):
  """Runs inference on an image."""
  # load image
  image_data = load_image(file_name)
  # load graph, which is stored in the default session
  results = run_graph(image_data, 0, input_layer, output_layer,
            3)
  print(results)
  if results[0]>0.3:
      return 1
  return 0
      

def filterImages():
    load_graph(graph)
    step=1
    x = []
    y = []
    i = 0
    for subdir, dirs ,files in os.walk(directory):
        for file in files:
            if i<=3:
                i+=1
                break
            if file.endswith(".jpg"):
                try:
                    fileName= os.path.join(subdir,file)
                    print(fileName)
                    toDelete= predictImage(fileName,graph)
                    if toDelete==1:
                        os.remove(fileName)
                    print("step: "+str(step))
                    step+=1
                except:
                    print("single error")
                    try:
                        os.remove(fileName)
                    except:
                        print("double error")

                                








