# multimodal-deep-learning-for-diaster-response
Code and Dataset for paper: "Damage Identification in Social Media Posts using Multimodal Deep Learning" authors: Hussein Mouzannar, Yara Rizk, Mariette Awad; to appear in ISCRAM 2018
# Requirements
Python 3.6, Tensorflow 1.40 (and all its dependencies)
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

