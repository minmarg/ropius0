## (C)2020 Mindaugas Margelevicius, Vilnius University
## adapted from Semantic Segmentation Suite
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

sys.path.insert(1, '/data/installed-software/ROPIUS0/bin')
sys.path.insert(1, '/data/installed-software/ROPIUS0/infer')
sys.path.insert(1, '/data/installed-software/Semantic-Segmentation-Suite')
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
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
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
basedir = '/home2/mindaugas/projects/data/pdb70_from_mmcif_200205__selection/'
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

# Load a previous checkpoint if desired
##{{MM
chkpathname = 'checkpoints_519'
chkpathname = 'checkpoints_519_tests'
chkpathname = 'checkpoints_519_normonly_from_scratch'
chkpathname = 'checkpoints_519_normonly_from_scratch__lr00005'
chkpathname = 'checkpoints_519_normonly_from_scratch__lr0001'
chkpathname = 'checkpoints_519_normonly_from_scratch__'+args.model+'_lr'+str(optlrate)
datetime_at_start = datetime.now().strftime("%Y%m%d%H%M%S")
model_checkpoint_name = chkpathname + "/latest_model_" + args.model + "_promage.ckpt"
##}}
##model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)
else:
    ##{{MM
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        model_variables = slim.get_model_variables('resnet_v2_101')
        init_op = tf.variables_initializer(model_variables)
        sess.run(init_op)
    ##}}


print("\n***** Begin training *****")
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

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
avg_val_loss_per_epoch = [] ##{{MM}}

TRAINSTEP = 32 ##{{MM}}
VALFULL = False ##{{MM}} Validation to be performed on full promages
VALSTEP = 0 #64 ##{{MM}}
VALBATCHSIZE = 32 ##{{MM}} max number of promages to perform simultaneous validation on

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)


def readData(i, j, step):
    """Collect a batch of images.
    """
    index = i * step + j
    id = id_list[index]
    ##{{MM
    name = os.path.basename(train_input_names[id]).rsplit('.', 2)[0]
    name_index = promages_train.get_image_index(name)
    input_image = promages_train.load_image(name_index)
    output_image, _ = promages_train.load_mask(name_index)
    image_shape = input_image.shape
    ##}}
    ##input_image = utils.load_image(train_input_names[id])
    ##output_image = utils.load_image(train_output_names[id])

    with tf.device('/cpu:0'):
        ##{{MM
        if TRAINSTEP < 1:
            input_image, window, scale, padding, crop = resize_image(input_image,
                min_dim=conf.IMAGE_MIN_DIM, max_dim=conf.IMAGE_MAX_DIM,
                min_scale=None, mode=conf.IMAGE_RESIZE_MODE)
            output_image = resize_mask(output_image, scale, padding, crop)
        else:
            _, _, _, _, crop = resize_image(input_image,
                min_dim=conf.IMAGE_MIN_DIM, max_dim=conf.IMAGE_MAX_DIM,
                min_scale=None, mode=conf.IMAGE_RESIZE_MODE)
        ##}}
        ##input_image, output_image = data_augmentation(input_image, output_image)


        # Prep the data. Make sure the labels are in one-hot format
        ##{{MM
        if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
        if NORMALIZE: input_image = input_image / conf.PIXEL_NORM
        ##}}
        ##input_image = np.float32(input_image) / 255.0

        ##{{MM:mask os already one-hot-coded}}
        ##output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

    return image_shape, crop, input_image, output_image


def pad64_image(image, mask=False):
    """Pad image with zeros so that its dimensions are multiples of 64
    """
    # return results in the same dtype
    image_dtype = image.dtype
    h, w = image.shape[:2]
    # Height
    if h % 64 > 0:
        max_h = h - (h % 64) + 64
        ## the origin is better for prediction:
        top_pad = 0 #(max_h - h) // 2
        bottom_pad = max_h - h - top_pad
    else:
        top_pad = bottom_pad = 0
    # Width
    if w % 64 > 0:
        max_w = w - (w % 64) + 64
        ## the origin is better for prediction:
        left_pad = 0 #(max_w - w) // 2
        right_pad = max_w - w - left_pad
    else:
        left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    if mask:
        image[:window[2],:window[3],:] = True
    return image.astype(image_dtype), window, padding



val_evals = 0
##{{MM: number of parallel workers
nparallelwrokers = 32
if nparallelwrokers < args.batch_size:
    nparallelwrokers = args.batch_size
##}}

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    ##{{MM
    num_iters = int(np.floor(len(id_list) / nparallelwrokers))
    ###num_iters = 0 ##<<<<<<<<<<<<<<<<<<<<<
    ##}}
    ##num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()

        ##inputoutput = Parallel(n_jobs=args.batch_size)(delayed(readData)(i, j, args.batch_size) for j in range(args.batch_size))
        inputoutput = Parallel(n_jobs=nparallelwrokers)(delayed(readData)(i, j, nparallelwrokers) for j in range(nparallelwrokers))
        image_shapes, image_crops, inputimages, outputimages = zip(*inputoutput)
        ##[input_image_batch.append(np.expand_dims(inputimages[imndx], axis=0)) for imndx in range(len(inputimages))]
        ##[output_image_batch.append(np.expand_dims(outputimages[imndx], axis=0)) for imndx in range(len(outputimages))]

        # Collect a batch of images
        ##nparts = nparallelwrokers // args.batch_size
        for k in range(0, nparallelwrokers, args.batch_size):

            input_image_batch = []
            output_image_batch = []

            if TRAINSTEP < 1:

                [input_image_batch.append(np.expand_dims(inputimages[imndx], axis=0)) for imndx in range(k,k+args.batch_size)]
                [output_image_batch.append(np.expand_dims(outputimages[imndx], axis=0)) for imndx in range(k,k+args.batch_size)]

                if args.batch_size == 1:
                    input_image_batch = input_image_batch[0]
                    output_image_batch = output_image_batch[0]
                else:
                    input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                    output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

                # Do the training
                _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
                current_losses.append(current)
                cnt = cnt + args.batch_size
                if cnt % 20 == 0:
                    string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                    utils.LOG(string_print)
                    st = time.time()

            else:
                for ki in range(k,k+args.batch_size):
                    loaded_image = inputimages[ki]
                    loaded_output = outputimages[ki]
                    ishape = image_shapes[ki]
                    icrop = image_crops[ki]
                    ndxi_beg = icrop[0]
                    ndxi_end = icrop[0] + icrop[2]
                    ndxj_beg = icrop[1]
                    ndxj_end = icrop[1] + icrop[3]
                    
                    while ndxi_end <= ishape[0]:
                        while ndxj_end <= ishape[1]:
                            input_image = loaded_image[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end, :]
                            output_image = loaded_output[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end, :]
                            input_image_batch.append(np.expand_dims(input_image, axis=0))
                            output_image_batch.append(np.expand_dims(output_image, axis=0))
                            if args.batch_size <= len(input_image_batch) or \
                               (ndxi_end >= ishape[0] and ndxj_end >= ishape[1] and ki+1 >= k+args.batch_size):

                                if len(input_image_batch) == 1:
                                    input_image_batch = input_image_batch[0]
                                    output_image_batch = output_image_batch[0]
                                else:
                                    input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                                    output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

                                # Do the training
                                _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
                                current_losses.append(current)
                                cnt = cnt + args.batch_size
                                if cnt % 20 == 0:
                                    string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                                    utils.LOG(string_print)
                                    st = time.time()

                                input_image_batch = []
                                output_image_batch = []

                            if ndxj_end < ishape[1]:
                                ndxj_beg += TRAINSTEP
                                ndxj_end = ndxj_beg + icrop[3]
                                if ndxj_end > ishape[1]:
                                    ndxj_end = ishape[1]
                                    ndxj_beg = ndxj_end - icrop[3]
                            else: break

                        if ndxi_end < ishape[0]:
                            ndxj_beg = 0
                            ndxj_end = icrop[3]
                            ndxi_beg += TRAINSTEP
                            ndxi_end = ndxi_beg + icrop[2]
                            if ndxi_end > ishape[0]:
                                ndxi_end = ishape[0]
                                ndxi_beg = ndxi_end - icrop[2]
                        else: break

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)
    ##{{MM
    print("\n\nEpoch = %d Mean_Loss = %.4f Time = %.2f\n"% (epoch, mean_loss, time.time()-st), flush=True)
    ##}}

    ##{{MM
    if epoch % args.checkpoint_step and epoch % args.validation_step and epoch+1 < args.num_epochs:
        continue
    ##}}

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%(chkpathname, epoch)):
        os.makedirs("%s/%04d"%(chkpathname, epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    ##{{MM}}if val_indices != 0 and epoch % args.checkpoint_step == 0:
    if epoch % args.checkpoint_step == 0 or epoch+1 >= args.num_epochs:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(chkpathname, epoch))

    ##{{MM
    if (epoch % args.validation_step and epoch+1 < args.num_epochs) or num_vals < 1:
        continue
    ##}}

    print("Performing validation")
    ##{{MM}}target=open("%s/%04d/val_scores.csv"%(chkpathname, epoch),'w')
    ##{{MM}}target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    ##{{MM
    val_loss = []
    input_image_batch = []
    output_image_batch = []
    gt_image_batch = []
    output_preds_batch = []
    ##}}

    # Do the validation on a small set of validation images
    for valndx, ind in enumerate(val_indices):

        ##{{MM
        name = os.path.basename(val_input_names[ind]).rsplit('.', 2)[0]
        name_index = promages_val.get_image_index(name)
        input_image = promages_val.load_image(name_index)
        gt, _ = promages_val.load_mask(name_index)
        gt_decoded_image = helpers.reverse_one_hot(gt)

        if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
        if NORMALIZE: input_image = input_image / conf.PIXEL_NORM

        if VALSTEP < 1 and not VALFULL:
            if VAL_RND_CROPS:
                input_image, window, scale, padding, crop = resize_image(input_image,
                        min_dim=conf.IMAGE_MIN_DIM, max_dim=conf.IMAGE_MAX_DIM,
                        min_scale=None, mode=conf.IMAGE_RESIZE_MODE)
            else:
                input_image = input_image[:args.crop_height,:args.crop_width,:]
            input_image = np.expand_dims(input_image, axis=0)
            if VAL_RND_CROPS:
                gt = resize_mask(gt, scale, padding, crop)
            else:
                gt = gt[:args.crop_height,:args.crop_width,:]
            gtencoded = np.expand_dims(gt, axis=0)
            gt = helpers.reverse_one_hot(gt)
            ##}}
            ##input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            ##gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            ##gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            # st = time.time()

            ##with tf.device('/gpu:1'): ##{{MM}}
            output_image = sess.run(network,feed_dict={net_input:input_image})
            ##{{MM
            val_loss_1 = sess.run(loss,feed_dict={net_input:input_image,net_output:gtencoded})
            val_loss.append(val_loss_1)
            ##}}


            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            ##{{MM}}out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

            ##{{MM}}file_name = utils.filepath_to_name(val_input_names[ind])
            ##{{MM}}target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            ##{{MM}}for item in class_accuracies:
            ##{{MM}}    target.write(", %f"%(item))
            ##{{MM}}target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            ##{{MM}}gt = helpers.colour_code_segmentation(gt, label_values)

            ##{{MM}}file_name = os.path.basename(val_input_names[ind])
            ##{{MM}}file_name = os.path.splitext(file_name)[0]
            ##{{MM}}cv2.imwrite("%s/%04d/%s_pred.png"%(chkpathname, epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            ##{{MM}}cv2.imwrite("%s/%04d/%s_gt.png"%(chkpathname, epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

        ##{{MM
        else:
            if VALSTEP < 1:
                sys.exit('Invalid step: VALSTEP= %d'%(VALSTEP))
            if VALFULL:
                loaded_image, window, _ = pad64_image(input_image)
                loaded_output, _, _ = pad64_image(gt, mask=True)
                gt_decoded_image = helpers.reverse_one_hot(loaded_output)
                crop_height = loaded_image.shape[0]
                crop_width = loaded_image.shape[1]
            else:
                loaded_image = input_image
                loaded_output = gt
                crop_height = args.crop_height
                crop_width = args.crop_width
            ndxi_beg = 0
            ndxi_end = crop_height
            ndxj_beg = 0
            ndxj_end = crop_width

            while ndxi_end <= loaded_image.shape[0]:
                ndxj_beg = 0
                ndxj_end = crop_width
                while ndxj_end <= loaded_image.shape[1]:
                    input_image_crop = loaded_image[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end, :]
                    output_image = loaded_output[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end, :]
                    gt_image = gt_decoded_image[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end]
                    input_image_batch.append(np.expand_dims(input_image_crop, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))
                    gt_image_batch.append(np.expand_dims(gt_image, axis=0))
                    if VALBATCHSIZE <= len(input_image_batch) or \
                       (ndxi_end >= loaded_image.shape[0] and ndxj_end >= loaded_image.shape[1] and valndx+1 >= len(val_indices)):

                        if len(input_image_batch) == 1:
                            input_image_batch = input_image_batch[0]
                            output_image_batch = output_image_batch[0]
                        else:
                            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

                        # Do validation
                        output_preds_batch, val_loss_1 = sess.run([network,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
                        val_loss.append(val_loss_1)

                        for vali in range(len(output_preds_batch)):
                            output_image = np.array(output_preds_batch[vali][:,:,:])
                            output_image = helpers.reverse_one_hot(output_image)
                            gt_image = gt_image_batch[vali][:,:]

                            accuracy, class_accuracies, prec, rec, f1, iou = \
                              utils.evaluate_segmentation(pred=output_image, label=gt_image, num_classes=num_classes)

                            scores_list.append(accuracy)
                            class_scores_list.append(class_accuracies)
                            precision_list.append(prec)
                            recall_list.append(rec)
                            f1_list.append(f1)
                            iou_list.append(iou)

                        print("A total of %d validation promage crops processed (promage %d)"%(len(output_preds_batch),valndx), flush=True)

                        input_image_batch = []
                        output_image_batch = []
                        gt_image_batch = []
                        output_preds_batch = []

                    if ndxj_end < loaded_image.shape[1]:
                        ndxj_beg += VALSTEP
                        ndxj_end = ndxj_beg + crop_width
                        if ndxj_end > loaded_image.shape[1]:
                            ndxj_end = loaded_image.shape[1]
                            ndxj_beg = ndxj_end - crop_width
                    else: break

                if ndxi_end < loaded_image.shape[0]:
                    ndxi_beg += VALSTEP
                    ndxi_end = ndxi_beg + crop_height
                    if ndxi_end > loaded_image.shape[0]:
                        ndxi_end = loaded_image.shape[0]
                        ndxi_beg = ndxi_end - crop_height
                else: break
        ##}}



    ##{{MM}}target.close()

    avg_val_loss_per_epoch.append(np.mean(val_loss))  ##{{MM}}
    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_scores_per_epoch.append(avg_score)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_iou_per_epoch.append(avg_iou)
    val_evals += 1

    ##{{MM
    print("\nAverage validation loss for epoch # %04d = %f"% (epoch, avg_val_loss_per_epoch[-1]))
    ##}}
    print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
    print("Average per class validation accuracies for epoch # %04d:"% (epoch))
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Validation precision = ", avg_precision)
    print("Validation recall = ", avg_recall)
    print("Validation F1 score = ", avg_f1)
    print("Validation IoU score = ", avg_iou, flush=True)



    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []


    ##{{MM:
    ##fig0, ax0 = plt.subplots(figsize=(11, 8))

    ##ax0.plot(range(len(avg_val_loss_per_epoch)), avg_val_loss_per_epoch)
    ##ax0.set_title("Average validation loss vs epochs")
    ##ax0.set_xlabel("Epoch")
    ##ax0.set_ylabel("Avg. val. loss")

    ##plt.savefig(os.path.join(chkpathname,'val_loss_vs_epochs_'+datetime_at_start+'.png'))

    ##plt.clf()
    ##}}

    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ##{{MM:
    ax1.plot(range(len(avg_scores_per_epoch)), avg_scores_per_epoch)
    ##}}##ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig(os.path.join(chkpathname,'accuracy_vs_epochs_'+datetime_at_start+'.png'))

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ##{{MM:
    ax2.plot(range(len(avg_loss_per_epoch)), avg_loss_per_epoch, label='train')
    ax2.plot(range(len(avg_val_loss_per_epoch)), avg_val_loss_per_epoch, label='validation')
    ax2.legend()
    ##}}##ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig(os.path.join(chkpathname,'loss_vs_epochs_'+datetime_at_start+'.png'))

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ##{{MM:
    ax3.plot(range(len(avg_iou_per_epoch)), avg_iou_per_epoch)
    ##}}##ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig(os.path.join(chkpathname,'iou_vs_epochs_'+datetime_at_start+'.png'))

    plt.clf()

