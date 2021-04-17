#!/usr/bin/env python

from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time ##{{MM}}, datetime
from datetime import datetime ##{{MM}}
import argparse
import random
import os, sys
import subprocess

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')
import multiprocessing
from joblib import Parallel, delayed

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'bin'))
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'infer'))
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'Semantic-Segmentation-Suite'))

from frontends import resnet_v2
from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

# private imports
from mrcnn.utils import resize_image, resize_mask
import promage4segm_519 as pmg

import time


# initialize seeds for RNGs
np.random.seed(int(time.time()))
tf.set_random_seed(int((time.time()-int(time.time()))*1e7))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=128, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=128, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=200, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="", help='The frontend you are using. See frontend_builder.py for supported models') ##ResNet101
parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use')
args = parser.parse_args()


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

##{{MM
##config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4})
config = tf.ConfigProto(device_count={'GPU':1},gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(args.gpu)),log_device_placement=False)
##}}
##config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

##{{MM
VAL_RND_CROPS = True ##random crops for validation
SUBT_MEAN = False
NORMALIZE = True #False
conf = pmg.PromageConfig()
net_input = tf.placeholder(tf.float32,shape=[None,None,None,conf.IMAGE_CHANNEL_COUNT])
##
if args.crop_height != args.crop_width:
    sys.exit("ERROR: Crop height and width should be equal.")
basedir = '/data/CASP14_datasets/'
conf.IMAGE_MIN_DIM = args.crop_height
trnset_prodir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_pro__training'
trnset_covdir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_cov__training'
trnset_pdbdir = basedir + 'pdb70_from_mmcif_200205__selection__pdb__training'
trnset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__SEMSEGM'
promages_train = pmg.PromageDataset(conf, train=True, keepinmemory=False)
promages_train.loadPromages(prodir=trnset_prodir, covdir=trnset_covdir,
        pdbdir=trnset_pdbdir, outdir=args.dataset)

valset_prodir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_pro__validation'
valset_covdir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_cov__validation'
valset_pdbdir = basedir + 'pdb70_from_mmcif_200205__selection__pdb__validation'
valset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__SEMSEGM'
promages_val = pmg.PromageDataset(conf, val=True, keepinmemory=True)
promages_val.loadPromages(prodir=valset_prodir, covdir=valset_covdir,
        pdbdir=valset_pdbdir, outdir=args.dataset)
# filter out too small samples
print("Filtering out too small training samples.")
train_input_names1 = []
train_output_names1 = []
for id,_ in enumerate(train_input_names):
    name = os.path.basename(train_input_names[id]).rsplit('.', 2)[0]
    name_index = promages_train.get_image_index(name)
    output_image, _ = promages_train.load_mask(name_index)
    if output_image.shape[0] < conf.IMAGE_PROC_MIN_DIM or \
       output_image.shape[1] < conf.IMAGE_PROC_MIN_DIM:
        continue
    train_input_names1.append(train_input_names[id])
    train_output_names1.append(train_output_names[id])
train_input_names = train_input_names1
train_output_names = train_output_names1
print("Total training samples, {}.".format(len(train_input_names)))
# same with the val set
print("Filtering out too small validation samples.")
val_input_names1 = []
val_output_names1 = []
for id,_ in enumerate(val_input_names):
    name = os.path.basename(val_input_names[id]).rsplit('.', 2)[0]
    name_index = promages_val.get_image_index(name)
    output_image, _ = promages_val.load_mask(name_index)
    if output_image.shape[0] < conf.IMAGE_PROC_MIN_DIM or \
       output_image.shape[1] < conf.IMAGE_PROC_MIN_DIM:
        continue
    val_input_names1.append(val_input_names[id])
    val_output_names1.append(val_output_names[id])
val_input_names = val_input_names1
val_output_names = val_output_names1
print("Total validation samples, {}.".format(len(val_input_names)))
##
#print("Calculating class weights: ...")
#class_weights = promages_train.class_stats()
#sum_class_weights = np.sum(class_weights)
#if sum_class_weights < 1:
#    sys.exit("ERROR: Invalid sum of class weights.")
##class_weights = ((np.max(class_weights)+100)-class_weights) / (np.sum((np.max(class_weights)+100)-class_weights))
##class_weights[0] = np.min(class_weights[1:])
#class_weights = class_weights / np.max(class_weights)
#min_wgt = np.max(class_weights) / 3
#for i in range(len(class_weights)):
#    if class_weights[i] < min_wgt: class_weights[i] = min_wgt
##class_weights = [1] * conf.NUM_CLASSES
##class_weights[0] = 1.5
#for i in range(40,len(class_weights)): class_weights[i] = 1.5
#with np.printoptions(precision=5, suppress=True):
#    print(class_weights)
#    print("Sum= {}".format(np.sum(class_weights)))
##}}


# Compute your softmax cross entropy loss
##net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

##{{MM
#weighted_logits = tf.multiply(network, class_weights)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=weighted_logits, labels=net_output))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=network, labels=net_output))
##}}
##loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

##{{MM
optlrate = 0.0001
#optlrate = 0.0005
#optlrate = 0.001
optdecay = 0.9
opt = tf.train.RMSPropOptimizer(learning_rate=optlrate, decay=optdecay).minimize(loss, var_list=[var for var in tf.trainable_variables()])
print("RMSPropOptimizer: Learning rate = {}; Decay = {}".format(optlrate,optdecay))
#admlrate = 0.001
#myoptimizer = tf.train.AdamOptimizer(learning_rate=admlrate)
#myoptimizer._create_slots(var_list=[var for var in tf.trainable_variables()])
#opt = myoptimizer.minimize(loss, var_list=[var for var in tf.trainable_variables()])
#print("AdamOptimizer: Learning rate = {}".format(admlrate))
##}}
##opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])


saver=tf.train.Saver(max_to_keep=1000)
##with tf.device('/gpu:1'): ##{{MM}}
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
##{{MM}}if init_fn is not None:
##{{MM}}    init_fn(sess)

print('Model structure:')
model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)



print("\nTraining data:")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")
