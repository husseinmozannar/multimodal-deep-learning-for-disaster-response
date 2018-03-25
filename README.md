# multimodal-deep-learning-for-diaster-response
Code and Dataset for paper: "Damage Identification in Social Media Posts using Multimodal Deep Learning" authors: Hussein Mouzannar, Yara Rizk, Mariette Awad; to appear in ISCRAM 2018
# Requirements
Python 3.6, Tensorflow 1.40 (and all its dependencies)

# Dataset

The multimodal dataset (image and text for each post) is collected from social media sites (Twitter and Instagram) and labeled by a group of 5 volunteers. The dataset was used in the aforementioned paper and is available for download only for academic purposes using the following link: 

https://drive.google.com/open?id=1lLhTpfYBFaYwlAVaH7J-myHuN8mdV595

Please cite our paper if you use this dataset in an academic publication.

# Image model Training
(these steps are better described at https://github.com/tensorflow/models/tree/master/research/slim)

The first step is to prepare the image data for tensorflow slim:

1- Organize image data in one parent directory in the following format: label1\image1.jpg ... imageN.jpg, label2\...

2- Build a slim dataset using the data: first make sure to include disaster and the updated dataset factory in your slim\datasets directory, then using build_image_data located in slim\models\research\inception\inception\data run:

python build_image_data.py --train_directory={TRAIN_DATASET} --validation_directory={VAL_DATASET} --labels_file={labels text file: each line: label1 \newline label2 ...}  --output_directory={TF_DATASET}

Now we can train our image-model:

python train_image_classifier.py   --train_dir={DIRECTORY TO PLACE MODEL FILES}  --dataset_dir={TF_DATASET} --dataset_name=disaster  --dataset_split_name=train --model_name=inception_v3  --checkpoint_path={IMAGENET CHECKPOINT FILE} 
--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits

After training you now need to produce a graph file to be able to use your model for prediction:

python export_inference_graph.py  --alsologtostderr --model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

python freeze_graph.py --input_graph=/tmp/inception_v3_inf_graph.pb  --input_checkpoint={DIRECTORY FOR MODEEL FILES\model.ckpt-{steps}}
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb  --output_node_names=InceptionV3/Predictions/Reshape_1
Now you can use frozen_inception_v3.pb.

# Text model training

We adapt the code from https://github.com/dennybritz/cnn-text-classification-tf to our dataset and add the ability to display a confusion matrix and predict single captions.

train.py produces for each model a folter containing graph data that can be automatically used later for prediction.

# Decision Fusion

The first step is to obtain predictions using our trained image and text models on the train/test/val sets:

1- get image representations and save them: use loadDataImage.py

2- get text representation and save them: use loadDataText.py: make sure to load the correct text model

You will have .pkl file for each predictions for easy reuse later on.

After you have have obtained your new "dataset", annDecision implements a simple neural network and computes accuracies, svm_knn implements a linear and guassian svm and knn models for decision fusion to get accuracies. Finally decisionRules implements a max decision rule and computes accuracies

# Feature Fusion

Same as for Decision fusion, obtain your new dataset of features using the same scripts (but different modules). annFeatures implements a deep neural network and svm_knn as before.

# Visual

We have an implementation of PCA and LDA to visualize our features in 2D, just load your data and run.
