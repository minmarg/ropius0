#!/usr/bin/env python

##
## (C)2020 Mindaugas Margelevicius, Vilnius University
##

import os
import sys
import fnmatch
import random
import math
import re
import time
import logging
import cv2
import math
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from scipy.sparse import dok_matrix
from Bio import Align
from Bio.Align import substitution_matrices
from optparse import OptionParser

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as mrclib
from mrcnn import visualize
from mrcnn.model import log

## private imports
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'bin'))
from getchainseq import getChainSequence
from getdist import getDistances

MYMODNAME = os.path.basename(sys.modules[__name__].__file__)

## ===========================================================================

description = "Prepare a promage dataset and initiate training."

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)
    parser.add_option("--make1", dest="make1pathname", default="", metavar="PATHNAME",
                      help="Make a promage given the name (without extension) "+
                           "of a profile and xcov file and exit")
    parser.add_option("--mask1", dest="mask1pathname", default="", metavar="PATHNAME",
                      help="Make a mask for the given pdb structure file and exit")
    parser.add_option("--makemasks", dest="makemasks", default="", metavar="PATHNAME",
                      help="Make masks for all pdb structures under the given pathname. "+
                           "Output directory will be created with entension .msk added. "+
                           "NOTE: Option make1 common to all the pdbs should be specified "+
                           "too. Links to the resulting promage will be created in a new "+
                           "directory with extension .pmg.")
    parser.add_option("--make", action="store_true", dest="make",
                      help="Make a dataset and exit")
    parser.add_option("--train", action="store_true", dest="train",
                      help="Train a Mask-RCNN model")
    parser.add_option("--infer", dest="pmginfer", default="",
                      help="Name of promage (w/o extension) to apply the "+
                           "model to", metavar="FILE")
    parser.add_option("--visual", action="store_true", dest="visual",
                      help="Visualize the weights of the most recent "+
                           "Mask-RCNN model")
    parser.add_option("--restart", action="store_true", dest="restart",
                      help="Reinitialize weights instead of "+
                           "continuing from the last checkpoint")
    parser.add_option("--stats", action="store_true", dest="stats",
                      help="Calculate class statistics for training set")
    parser.add_option("-l", "--lrate", dest="lrate", default=0.001, 
                      type=float,
                      help="Learning rate; default=%default", metavar="RATE")
    ret = 1
    (options, args) = parser.parse_args()
    if options.lrate <= 0 or options.lrate >= 1.0:
        sys.stderr.write("\nERROR: Invalid learning rate.\n")
        ret = 0
    return ret, options


## ===========================================================================
##
def getNProfileAttrs():
    """Get the number of profile attributes.
    """
    return 1+1+20+7+3+1+2+21+3


def getNXcovfileAttrs():
    """Get the number of xcov file attributes.
    """
    return 401 # 20*20 + 1 (MI)


## ===========================================================================
## Configuration
##
class PromageConfig():
    """Configuration for training on a profile image dataset.
    Derives from the base Config class and overrides values specific to
    profile image dataset.
    """
    # configuration name
    NAME = "promages"
    # configuration application type
    TYPE = "training"

    # Train on GPU_COUNT GPUs and IMAGES_PER_GPU promages per GPU. Promages
    #can be large, the number of promages assigned to each GPU cannot be large.
    # Batch size is IMAGES_PER_GPU * GPU_COUNT.
    GPU_COUNT = 2
    # A 12GB GPU can typically handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000 #1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 200 #50


    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256


    # Number of classes (including background)
    NUM_CLASSES = 1 + 63  # background + 63 distance bins (2,3,...,64)
    FST_CLASS = 2 # name (distance) of the first/starting class
    LST_CLASS = 64 # name (distance) of the last class


    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) # anchor side in pixels

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 4000 #6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = 'crop' #'pad_and_crop' #'crop' #"pad64"
    IMAGE_PROC_MIN_DIM = 128 ##MM_edit #minimum image dimension that allows for processing
    IMAGE_MIN_DIM = 128 #64 #None
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = None
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = getNXcovfileAttrs() + getNProfileAttrs() * 2 # NOTE: number of promage features/channels
    #IMAGE_CHANNEL_COUNT = getNXcovfileAttrs()
    #IMAGE_CHANNEL_COUNT = 1 + getNProfileAttrs() * 2
    #IMAGE_CHANNEL_COUNT = 1 + 10
    #IMAGE_CHANNEL_COUNT = 1 + 2
    # Image mean (RGB)
    MEAN_PIXEL = np.concatenate(( np.repeat(0., 400),
                                  [1.0],
                                  np.tile(
                                    np.concatenate((
                                      np.array([245, 10], dtype=np.float32), 
                                      np.repeat(0.05, 20), np.repeat(0.25, 7),
                                      np.repeat(4., 3), [0.5], [0.5, 600],
                                      [-18.5, 15.8], np.repeat(0., 19),
                                      np.repeat(0.5, 3) )), 2) )).astype(np.float32)
    #MEAN_PIXEL = np.zeros(IMAGE_CHANNEL_COUNT, dtype=np.float32)
    # normalization constant for a pixel
    PIXEL_NORM = np.concatenate(( np.repeat(1., 400),
                                  [1.0],
                                  np.tile(
                                    np.concatenate((
                                      np.array([1024, 20], dtype=np.float32), 
                                      np.repeat(1., 20), np.repeat(1., 7),
                                      np.repeat(14., 3), [1.], [1., 1260],
                                      [20., 20.], np.repeat(4., 19),
                                      np.repeat(1., 3) )), 2) )).astype(np.float32)


    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000 #100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1000 #100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 0.,#1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 0.,#1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = None #False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0



## ===========================================================================
## Configuration for inference
##
class InferencePromageConfig(PromageConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # configuration application type
    TYPE = "inference"
    # limits
    IMAGE_RESIZE_MODE = 'pad64' #'pad_and_crop' #'crop' #"pad64"
    IMAGE_PROC_MIN_DIM = 1 ##MM_edit #minimum image dimension that allows for processing
    #IMAGE_MIN_DIM = 128 #64 #None
    #IMAGE_MAX_DIM = 2048



## ===========================================================================
## General processing routines and data for promages
##

ALPH_RES = 'ARNDCQEGHILKMFPSTWYVBZJX'
ALPH_res = 'arndcqeghilkmfpstwyvbzjx'


def getValues(name, position, strlist, scale, expnum, skipndxs=None):
    """Convert each entry in the list strlist to a float32 value and
    scale it; expnum, expected number of entries in the list;
    skipndxs, list of the indices to skip.
    """
    mydefname = 'getValues: '
    if len(strlist) != expnum:
        sys.exit("ERROR: %sInvalid file format at pos %d: " %
            (mydefname,position) +
            "Invalid field count: %d vs. %d (%s)" %
            (expnum,len(strlist),os.path.basename(name)))
    if skipndxs is None:
        vals = np.array([None] * len(strlist))
    else:
        vals = np.array([None] * (len(strlist)-len(skipndxs)))
    j = 0
    for i, strval in enumerate(strlist):
        if skipndxs is not None and i in skipndxs: continue
        try:
            vals[j] = np.float32(int(strval) / scale) \
              if scale != 1 else np.float32(strval)
            j += 1
        except ValueError:
            sys.exit("ERROR: {}Invalid file format at pos {} ({})".
                format(mydefname,position,os.path.basename(name)))
    return vals


def readProfile(profile_pathname):
    """Read a profile an return all profile information wrapped in a list
    """
    mydefname = 'readProfile: '
    if not os.path.isfile(profile_pathname):
        sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)

    proname = os.path.basename(profile_pathname)
    numattrs = getNProfileAttrs() # number of profile attributes/fields at each position
    lastfld = 0 # last processed profile field

    with open(profile_pathname) as fp:
        position = -1
        recnum = 0
        for line in fp:
            #line = fp.readline()
            flds = re.split(r'\s+', line.rstrip())
            if flds[0] == 'LEN:':
                try:
                    prolen = int(flds[1])
                    if prolen < 1:
                        sys.exit("ERROR: {}Invalid file format: Length non-positive ({})".
                            format(mydefname,proname))
                except ValueError:
                    sys.exit("ERROR: {}Invalid file format: Invalid length ({})".
                        format(mydefname,proname))
                seq = np.empty(prolen, dtype=np.unicode_)
                pro = np.empty([prolen, numattrs], dtype=np.float32, order='C')
                continue
            if flds[0] == 'SCALE:':
                try:
                    scale = int(flds[1])
                except ValueError:
                    sys.exit("ERROR: {}Invalid file format: Invalid scale ({})".
                        format(mydefname,proname))
                continue
            if flds[0].isdecimal() and \
               flds[1].isalpha() and len(flds[1]) == 1:
               #flds[1].isascii() and \
                ## residue and target probabilities
                position += 1
                rescode = 0
                try:
                    rescode = ALPH_RES.index(flds[1])
                except ValueError:
                    sys.exit("ERROR: {}Invalid file format: Invalid residue at pos {} ({})".
                        format(mydefname,position,proname))
                nents = 20
                seq[position] = flds[1]
                pro[position,:2+nents] = np.concatenate((
                    [np.float32(flds[0]),rescode],
                    getValues(profile_pathname, position, flds[2:], scale, nents)))
                lastfld = 2+nents
                recnum = 1
            if position < 0:
                continue
            if not flds[0]:
                ## information for a position follows
                if recnum == 1:
                    ## transition probabilities
                    nents = 9
                    pro[position,lastfld:lastfld+nents-2] = \
                      getValues(profile_pathname, position, flds[1:], scale, nents, [5,7])
                    lastfld += nents-2
                    recnum += 1
                elif recnum == 2:
                    ## effective numbers of observations
                    nents = 3
                    pro[position,lastfld:lastfld+nents] = \
                      getValues(profile_pathname, position, flds[1:], 1000, nents)
                    lastfld += nents
                    recnum += 1
                elif recnum == 3:
                    ## CLUSTERS: HDP1: background posterior probability
                    nents = 2
                    pro[position,lastfld:lastfld+nents-1], noppps = \
                      getValues(profile_pathname, position, flds[1:], scale, nents)
                    if noppps <= 0:
                        sys.exit("ERROR: {}Invalid file format: Invalid # posteriors at pos {} ({})".
                            format(mydefname,position,proname))
                    lastfld += nents-1
                    recnum += 1
                elif recnum == 4:
                    ## CLUSTERS: HDP1: largest posterior predictive probability
                    nents = len(flds[1:])
                    pro[position,lastfld:lastfld+1] = \
                      getValues(profile_pathname, position, flds[1:], scale, nents, range(1,nents))
                    lastfld += 1
                    recnum += 1
                elif recnum == 5:
                    ## CLUSTERS: HDP1: index of the cluster corresponding to the largest posterior
                    nents = len(flds[1:])
                    pro[position,lastfld:lastfld+1] = \
                      getValues(profile_pathname, position, flds[1:], 1, nents, range(1,nents))
                    lastfld += 1
                    recnum += 1
                elif recnum == 6:
                    ## context vector
                    if flds[1] != 'CV:':
                        sys.exit("ERROR: {}Invalid file format: Invalid context vector at pos {} ({})".
                            format(mydefname,position,proname))
                    nents = 22
                    pro[position,lastfld:lastfld+nents-1] = \
                      getValues(profile_pathname, position, flds[2:], scale, nents, [2])
                    lastfld += nents-1
                    recnum += 1
                elif recnum == 7:
                    ## secondary structure
                    if not flds[1].startswith('SS:'):
                        sys.exit("ERROR: {}Invalid file format: Invalid secondary structure "+
                            "information at pos {} ({})".
                            format(mydefname,position,proname))
                    nents = 3
                    pro[position,lastfld:lastfld+nents] = \
                      getValues(profile_pathname, position, flds[2:], scale, nents)
                    lastfld += nents
                    recnum += 1
                else:
                    sys.exit("ERROR: {}Invalid file format: Unexpected information at pos {} ({})".
                        format(mydefname,position,proname))

    return {'seq': seq, 'pro': pro}


def readXCovfile(covfile_pathname):
    """Read a xcov file an return the information representing the upper
    triangle of the 2D profile square matrix saved in 3D array.
    """
    mydefname = 'readXCovfile: '
    if not os.path.isfile(covfile_pathname):
        sys.exit('ERROR: '+mydefname+'XCov file not found: '+covfile_pathname)

    covname = os.path.basename(covfile_pathname)
    numattrs = getNXcovfileAttrs() # number of xcov file attributes/fields at each pair of positions

    with open(covfile_pathname) as fp:
        position = 0
        p2 = -1 # along the other profile square matrix dimension
        for line in fp:
            flds = re.split(r'\s+', line.rstrip())
            if flds[0].startswith('#'): continue
            if p2 < 0 and \
               flds[0] == 'Length=' and flds[2] == 'xcov_size=':
                try:
                    prolen = int(flds[1])
                    if prolen < 1:
                        sys.exit("ERROR: {}Invalid file format: Length non-positive ({})".
                            format(mydefname,covname))
                except ValueError:
                    sys.exit("ERROR: {}Invalid file format: Invalid length ({})".
                        format(mydefname,covname))
                try:
                    xcovsize = int(flds[3])
                    if xcovsize != numattrs:
                        sys.exit("ERROR: {}Invalid file format: Wrong xcov size ({})".
                            format(mydefname,covname))
                except ValueError:
                    sys.exit("ERROR: {}Invalid file format: Invalid xcov size ({})".
                        format(mydefname,covname))
                xcov = np.empty([prolen, prolen, numattrs], dtype=np.float32, order='C')
                continue
            # data follows
            p2 += 1
            if p2 >= prolen:
                position += 1
                p2 = position
                if position >= prolen:
                    sys.exit(
                        "ERROR: {}Invalid file format: Out of profile matrix dimensions ({})".
                        format(mydefname,covname))
            try:
                if int(flds[0]) != position+1 and int(flds[1]) != p2+1:
                    sys.exit("ERROR: {}Invalid file format: Invalid indices at {} ({})".
                        format(mydefname,position,covname))
            except ValueError:
                sys.exit("ERROR: {}Invalid file format: Invalid indices at {} ({})".
                    format(mydefname,position,covname))
            xcov[position,p2,:numattrs] = \
              getValues(covfile_pathname, position, flds[2:], 1, numattrs)

    return xcov
            


def makePromage(profile_pathname, covfile_pathname):
    """Make a profile image given the full pathnames of a profile and its 
    cross-covariance matrices.
    """
    mydefname = 'makePromage: '
    if not os.path.isfile(profile_pathname):
        sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)
    if not os.path.isfile(covfile_pathname):
        sys.exit('ERROR: '+mydefname+'Xcov file not found: '+covfile_pathname)


## ===========================================================================
## Dataset definition
##
class PromageDataset(utils.Dataset):
    """Transform input profile and profile cross-covariance files to promage
    format and make a promage dataset.
    """
    mysource = 'promages'
    proext = '.pro'
    covext = '.cov'
    pdbext = '.ent'
    pmgext = '.pmg' ## extension of preprocessed promage file in binary format
    mskext = '.msk' ## file extension of preprocessed promage mask with its class ids in binary format
    npzext = '.npz' ## extension added by numpy when saving data in binary format


    def __init__(self, config, class_map=None, train=False, val=False, keepinmemory=False):
        super().__init__(class_map)
        self.config = config
        self.promage_names = []
        self.pathpro = None
        self.pathcov = None
        self.pathpdb = None
        self.pathpmg = None

        self.keepinmemory = keepinmemory
        self.train = train
        self.val = val
        self.pmgsubdir = ''
        self.msksubdir = ''
        self.classdict = 'class_dict.csv'

        if self.train:
            self.pmgsubdir = 'train'
            self.msksubdir = 'train_labels'
        elif self.val:
            self.pmgsubdir = 'val'
            self.msksubdir = 'val_labels'

        self.prepared_promages = [] # promages
        self.prepared_masks = [] # masks for each promage
        self.prepared_classids = [] # class ids for each mask


    def getFiles(self, directory, ext='', remext=True):
        """Read directory contents; ext, file extension;
        remext, remove extension.
        """
        files = []
        if not os.path.isdir(directory):
            sys.exit('ERROR: Directory not found: '+ directory)

        with os.scandir(directory) as entries:
            for entry in entries:
                #if entry.is_file() and (fnmatch.fnmatch(entry, '*2GNX_A' + ext) or fnmatch.fnmatch(entry, '*4BEG_A' + ext)):
                #if entry.is_file() and (fnmatch.fnmatch(entry, '*5J*' + ext) or fnmatch.fnmatch(entry, '*4BEG_A' + ext)):
                if entry.is_file() and fnmatch.fnmatch(entry, '*' + ext):
                    if remext:
                        files.append(os.path.splitext(entry.name)[0]) ##add filename only
                    else: files.append(entry.name)
        return files


    def getDstClass(self, distance):
        """Get the class for a given distance.
        distance, the distance given.
        """
        return int(round(distance, 0)) # round to the nearest integer


    def getSetSrtRndDistances(self, dstarray):
        """Get the set of sorted rounded observed distances.
        dstarray, array of distances.
        """
        # add the first class which will always be identified between the 
        # residues of the same position (self-distance)
        dstset = {self.config.FST_CLASS}
        [dstset.add(self.getDstClass(d)) for d in dstarray] # round to the nearest integer
        return sorted(dstset)


    def getClassIdsFroDistances(self, dstarray):
        """Get the numpy array of class ids for given distances.
        dstarray, array of appropriately rounded distances.
        """
        class_ids = []
        try:
            class_ids = np.array([self.class_names.index(int(d)) for d in dstarray])
        except ValueError:
            return 0, []
        return 1, class_ids




    def makeandsavePromageWithoutMask(self, index, name):
        """Make a promage corresponding to the given name for data at 
        location index in the dataset and write it to file.
        Mask is not made and written.
        """
        pmgpathname = os.path.join(self.pathpmg, self.pmgsubdir, name)
        if os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            return

        self.preparePromageWithoutMask(index)
        if not os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            np.savez_compressed(pmgpathname+self.pmgext,
                     promage=self.prepared_promages[index])
        # release resources
        self.prepared_promages[index] = None


    def makePromage1(self, procovfullpathname, outdir=None):
        """Read a profile and profile cross-covariances (filenames must match) to 
        define an output promage. Preprocess inputs and write them to a file in 
        outdir in binary format. procovfullpathname, full pathname of the 
        profile and xcov file without extensions.
        """
        mydefname = 'makePromage1: '
        source = self.mysource

        datadirname = os.path.abspath(os.path.dirname(procovfullpathname))
        databasename = os.path.basename(procovfullpathname)

        if outdir is None:
            outdir = datadirname

        # Add promages
        self.add_image(source, image_id=0, path=None, name=databasename)
        # Add classes
        class1 = self.config.FST_CLASS # name of the starting class
        for i in range(1,self.config.NUM_CLASSES):
            # the first class (0) is reserved for background
            self.add_class(source, i, str(i+class1-1))

        self.promage_names = [databasename]
        self.pathpro = datadirname
        self.pathcov = datadirname
        self.pathpdb = datadirname
        self.pathpmg = outdir

        self.prepared_promages = [None]
        self.prepared_masks = [None]
        self.prepared_classids = [None]

        self.prepare()

        if self.pathpmg and not os.path.exists(self.pathpmg):
            os.mkdir(self.pathpmg)

        # make and save
        pdbfile_pathname = os.path.join(self.pathpdb, databasename+self.pdbext)

        if os.path.isfile(pdbfile_pathname):
            self.makeandsavePromage(0, databasename)
        else:
            sys.stderr.write('PDB file not found, making mask omitted.\n')
            self.makeandsavePromageWithoutMask(0, databasename)





    def makeMask1(self, pdbfullpathname, outdir=None):
        """Read a pdb file and make a mask for it. Write the mask to a file in 
        outdir in binary format. pdbfullpathname, full pathname of the pdb file.
        """
        mydefname = 'makeMask1: '
        source = self.mysource

        datadirname = os.path.abspath(os.path.dirname(pdbfullpathname))
        databasename = os.path.basename(pdbfullpathname)

        if outdir is None:
            outdir = datadirname

        # Add promages
        self.add_image(source, image_id=0, path=None, name=databasename)
        # Add classes
        class1 = self.config.FST_CLASS # name of the starting class
        for i in range(1,self.config.NUM_CLASSES):
            # the first class (0) is reserved for background
            self.add_class(source, i, str(i+class1-1))

        self.promage_names = [databasename]
        self.pathpro = datadirname
        self.pathcov = datadirname
        self.pathpdb = datadirname
        self.pathpmg = outdir

        self.prepared_promages = [None]
        self.prepared_masks = [None]
        self.prepared_classids = [None]

        self.prepare()

        if self.pathpmg and not os.path.exists(self.pathpmg):
            os.mkdir(self.pathpmg)

        # make and save
        mskpathname = os.path.join(self.pathpmg, databasename)
        self.prepareMask(0, pdbfullpathname)

        if not os.path.isfile(mskpathname+self.mskext+self.npzext):
            np.savez_compressed(mskpathname+self.mskext,
                     promask=self.prepared_masks[0])
                     ##classids=self.prepared_classids[index])
        # release resources
        self.prepared_masks[0] = None
        self.prepared_classids[0] = None





    def makeandsaveMaskOnly(self, index, pdbfulldirname, pdbname, length, outputmskdir, outputpmgdir, pmgfile_pathname):
        """Make a mask for a pdb file, save it to a file and 
        create a link to the corresponding promage file;
        length, the (expected) length of all pdb structures.
        """
        if not os.path.isfile(pmgfile_pathname):
            sys.exit('ERROR: Promage file not found: %s'%(pmgfile_pathname))

        pmgfile_basename = pdbname+self.pmgext+self.npzext
        dstpmgfile_pathname = os.path.join(outputpmgdir, pmgfile_basename)
        #if not os.path.isfile(dstpmgfile_pathname):
        #    os.symlink(pmgfile_pathname, dstpmgfile_pathname)

        pdbfile_pathname = os.path.join(pdbfulldirname, pdbname)
        mskpathname = os.path.join(outputmskdir, pdbname)
        if os.path.isfile(mskpathname+self.mskext+self.npzext):
            return

        self.prepareMask(index, pdbfile_pathname, length )
        if not os.path.isfile(mskpathname+self.mskext+self.npzext):
            np.savez_compressed(mskpathname+self.mskext,
                     promask=self.prepared_masks[index])
                     ##classids=self.prepared_classids[index])
        # release resources
        self.prepared_masks[index] = None
        self.prepared_classids[index] = None


    def makeMasksOnly(self, pdbfulldirname, outdir=None):
        """Make masks for all pdb structures under the pathname pdbfulldirname.
        If not specified, output directory will be created with entension .msk added.
        NOTE: Links to the promage made in the preceding step will be created in a 
        new directory with extension .pmg.
        """
        mydefname = 'makeMasksOnly: '
        if not os.path.exists(pdbfulldirname):
            sys.exit('ERROR: Directory not found: %s'%(pdbfulldirname))

        name = self.promage_names[0]
        profile_pathname = os.path.join(self.pathpro, name+self.proext)

        pmgpathname = os.path.join(self.pathpmg, self.pmgsubdir, name)
        pmgfile_pathname = pmgpathname+self.pmgext+self.npzext

        if not os.path.isfile(profile_pathname):
            sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)

        if not os.path.isfile(pmgfile_pathname):
            sys.exit('ERROR: Promage to have been made not found: %s'%(pmgfile_pathname))

        prodct = readProfile(profile_pathname)
        length = len(prodct['pro'])

        outputmskdir = pdbfulldirname.strip(os.sep) + '.msk' if pdbfulldirname else 'msk'
        outputpmgdir = pdbfulldirname.strip(os.sep) + '.pmg' if pdbfulldirname else 'pmg'

        if not os.path.exists(outputmskdir):
            os.mkdir(outputmskdir)

        #if not os.path.exists(outputpmgdir):
        #    os.mkdir(outputpmgdir)

        pdbfiles = self.getFiles(pdbfulldirname, remext=False)
        npdbfiles = np.shape(pdbfiles)[0]
        print('[{}] {} structures read.'.format(MYMODNAME, npdbfiles))

        self.prepared_masks = [None] * npdbfiles
        self.prepared_classids = [None] * npdbfiles

        # parallelize
        n_cores = multiprocessing.cpu_count()//2 ##use physical cores
        Parallel(n_jobs=n_cores)(
            delayed(self.makeandsaveMaskOnly)(i, pdbfulldirname, pdbfiles[i], length, 
                    outputmskdir, outputpmgdir, pmgfile_pathname) 
                for i in range(npdbfiles))





    def makeandsavePromage(self, index, name):
        """Make a promage corresponding to the given name for data at 
        location index in the dataset and write it to file.
        """
        pmgpathname = os.path.join(self.pathpmg, self.pmgsubdir, name)
        mskpathname = os.path.join(self.pathpmg, self.msksubdir, name)
        if os.path.isfile(pmgpathname+self.pmgext+self.npzext) and \
           os.path.isfile(mskpathname+self.mskext+self.npzext):
            return

        self.preparePromage(index)
        if not os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            np.savez_compressed(pmgpathname+self.pmgext,
                     promage=self.prepared_promages[index])
        if not os.path.isfile(mskpathname+self.mskext+self.npzext):
            np.savez_compressed(mskpathname+self.mskext,
                     promask=self.prepared_masks[index])
                     ##classids=self.prepared_classids[index])
        # release resources
        self.prepared_promages[index] = None
        self.prepared_masks[index] = None
        self.prepared_classids[index] = None


    def makeandsaveClassDict(self):
        """Make the class dictionary.
        """
        dctpathname = os.path.join(self.pathpmg, self.classdict)
        if os.path.isfile(dctpathname):
            return

        with open(dctpathname,'wt') as fp:
            print('name,r,g,b', file=fp)
            print('Background,255,255,255', file=fp)
            col = 0
            # class ids arranged sequentially
            for d in range(self.config.FST_CLASS,self.config.LST_CLASS+1):
                print('{},{},{},{}'.
                    format(self.class_names[self.class_names.index(str(int(d)))],col,col,col),
                    file=fp)
                col = (col + 3) % 255


    def loadPromage1(self, pmgpathname):
        """Read the promage whose name is given without extension. 
        """
        mydefname = 'loadPromage1: '
        source = self.mysource
        # Add classes
        class1 = self.config.FST_CLASS # name of the starting class
        for i in range(1,self.config.NUM_CLASSES):
            # the first class (0) is reserved for background
            self.add_class(source, i, str(i+class1-1))

        if not os.path.isfile(pmgpathname+self.pmgext+self.npzext) and \
           not os.path.isfile(pmgpathname+self.mskext+self.npzext):
            sys.exit("ERROR: {}Promage not found ({})".
                format(mydefname,pmgpathname))

        pmgdirname = os.path.dirname(pmgpathname)
        pmgbasename = os.path.basename(pmgpathname)

        self.add_image(source, image_id=0, path=None, name=pmgbasename)

        self.promage_names = [pmgbasename]
        self.pathpmg = pmgdirname

        print('[{}] 1 promage read.'.format(MYMODNAME))

        self.prepared_promages = [None]
        self.prepared_masks = [None]
        self.prepared_classids = [None]

        self.prepare()



    def loadPromages(self, prodir, covdir, pdbdir, outdir=None, make=False):
        """Read the list of profiles, profile cross-covariances, and target pdb 
        structures (filenames must match) to define a dataset. If write is True,
        preprocess inputs and outputs and write them to files in outdir in 
        binary format for speeding up loading of promiges. If write is False,
        preprocessing will take place on the fly in load_image().
        """
        mydefname = 'loadPromages: '
        source = self.mysource
        # Add classes
        class1 = self.config.FST_CLASS # name of the starting class
        for i in range(1,self.config.NUM_CLASSES):
            # the first class (0) is reserved for background
            self.add_class(source, i, str(i+class1-1))

        # read files and check count and filename consistencies
        profiles = self.getFiles(prodir, self.proext)
        covfiles = self.getFiles(covdir, self.covext)
        pdbfiles = self.getFiles(pdbdir, self.pdbext)

        if np.shape(profiles)[0] != np.shape(covfiles)[0] or \
           np.shape(profiles)[0] != np.shape(pdbfiles)[0] or \
           np.shape(profiles)[0] < 1:
            sys.exit('ERROR: '+mydefname+'Inconsistent data.')

        for pf in profiles:
            if not pf in covfiles:
                sys.stderr.write('ERROR: '+mydefname+'Missing data: '+pf+'\n')
                sys.exit()
        for pf in profiles:
            if not pf in pdbfiles:
                sys.stderr.write('ERROR: '+mydefname+'Missing data: '+pf+'\n')
                sys.exit()

        npromages = np.shape(profiles)[0]

        print('[{}] {} data entries read.'.format(MYMODNAME, npromages))

        # Add promages
        for i in range(npromages):
            self.add_image(source, image_id=i, path=None,
                name=profiles[i])

        self.promage_names = profiles
        self.pathpro = prodir
        self.pathcov = covdir
        self.pathpdb = pdbdir
        self.pathpmg = outdir

        self.prepared_promages = [None] * npromages
        self.prepared_masks = [None] * npromages
        self.prepared_classids = [None] * npromages

        self.prepare()

        if self.pathpmg is None:
            return

        if not os.path.exists(self.pathpmg):
            os.mkdir(self.pathpmg)

        if self.train: self.makeandsaveClassDict()

        if not make: return

        dataready = 1
        for i in range(npromages):
            name = profiles[i]
            pmgpath = os.path.join(self.pathpmg, self.pmgsubdir)
            pmgpathname = os.path.join(pmgpath, name)
            if not os.path.exists(pmgpath):
                os.mkdir(pmgpath)
            mskpath = os.path.join(self.pathpmg, self.msksubdir)
            mskpathname = os.path.join(mskpath, name)
            if not os.path.exists(mskpath):
                os.mkdir(mskpath)
            if os.path.isfile(pmgpathname+self.pmgext+self.npzext) and \
               os.path.isfile(mskpathname+self.mskext+self.npzext):
                continue
            dataready = 0

        if dataready:
            return

        # parallel version
        n_cores = multiprocessing.cpu_count()//2 ##use physical cores
        Parallel(n_jobs=n_cores)(delayed(self.makeandsavePromage)(i, profiles[i]) for i in range(npromages))

#        # serial version
#        for i in range(npromages):
#            name = profiles[i]
#            pmgpathname = os.path.join(self.pathpmg, name)
#            if os.path.isfile(pmgpathname+self.pmgext+self.npzext) and \
#               os.path.isfile(pmgpathname+self.mskext+self.npzext):
#                continue
#            self.preparePromage(i)
#            np.savez_compressed(pmgpathname+self.pmgext,
#                     promage=self.prepared_promages[i])
#            np.savez_compressed(pmgpathname+self.mskext,
#                     promask=self.prepared_masks[i],
#                     classids=self.prepared_classids[i])
#            # release resources
#            self.prepared_promages[i] = None
#            self.prepared_masks[i] = None
#            self.prepared_classids[i] = None


    def class_stats(self, doprint=False):
        """Get class statistics over all training samples.
        """
        mydefname = 'class_stats: '
        num_images = len(self.image_info)
        nclasses = self.config.NUM_CLASSES
        clscounts = [0] * nclasses
        for iid in range(num_images):
            promask, _ = self.load_mask(iid)
            if promask is None:
                name = self.image_info[iid]['name']
                sys.exit('ERROR: {}Promage id {} ({}) is empty.'.format(mydefname,iid,name))
            if promask.shape[0] < self.config.IMAGE_PROC_MIN_DIM or \
               promask.shape[1] < self.config.IMAGE_PROC_MIN_DIM:
                continue
            clscounts += np.sum(promask, axis=(0,1))
        if doprint:
            total = np.sum(clscounts)
            if total < 1:
                sys.exit('ERROR: {}Total number of class observations is 0.'.format(mydefname))
            # print statistics
            print('ID  Name  Count    Fraction')
            print('{:3} {:3} {:9} {:6.4}'.format(0, 'B', clscounts[0], clscounts[0]/total))
            [print('{:3} {:3} {:9} {:6.4}'.format(i, self.class_names[i], clscounts[i], clscounts[i]/total)) for i in range(1,nclasses)]
        return clscounts



    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Pass to the parent class's function once images are encountered to 
        be not in this dataset.
        """
        info = self.image_info[image_id]
        if info['source'] == self.mysource:
            return info['name']
        else:
            super(self.__class__).image_reference(self, image_id)


    def get_image_index(self, image_name):
        return next(info for info in self.image_info if info['name'] == image_name)['id']


    def load_image(self, image_id):
        """Load the specified image and return a [H,W,C] Numpy array.
        If a dataset object was created with the preprocessing option of
        writing prepared promages to files, then the promage is loaded 
        from file. Otherwise, the promage is created from input files,
        profile, xcov, and pdb files.
        """
        mydefname = 'load_image: '
        info = self.image_info[image_id]
        name = info['name']
        if self.prepared_promages[image_id] is not None:
            promage = self.prepared_promages[image_id]
        elif self.pathpmg is not None:
            pmgfilename = os.path.join(self.pathpmg, self.pmgsubdir, name+self.pmgext+self.npzext)
            if not os.path.isfile(pmgfilename):
                sys.exit('ERROR: '+mydefname+'File not found: '+pmgfilename)
            with np.load(pmgfilename) as pmgfile_loaded:
                promage = pmgfile_loaded['promage']
                if self.keepinmemory:
                    self.prepared_promages[image_id] = promage
        else:
            self.preparePromage(image_id)
            promage = self.prepared_promages[image_id]

#        efflen = len(promage)
#        with np.printoptions(precision=3, suppress=True):
#            print(promage[0,1])
#            print(promage[1,0])
#            print(promage[10,10])
#            print(promage[efflen-1,efflen-2])

        #return promage[:,:,(400,402,461)]
        #return promage[:,:,(400,400,400)]
        return promage#[:,:,:401]#[:,:,(400,*range(401,403),*range(457,460),*range(460,462),*range(516,519))]


    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Load 
        instance masks and return them in the form of an array of 
        binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.

        Note: masks and class ids are loaded from file if a dataset object
        has been created with the preprocessing option. Otherwise, it is 
        assumed that the promage and masks have been prepared during a call
        to load_image and saved in the object's buffers.
        """
        mydefname = 'load_mask: '
        info = self.image_info[image_id]
        name = info['name']
        if self.prepared_masks[image_id] is not None:
           promask = self.prepared_masks[image_id]
           class_ids = self.prepared_classids[image_id]
        elif self.pathpmg is not None:
            mskfilename = os.path.join(self.pathpmg, self.msksubdir, name+self.mskext+self.npzext)
            if not os.path.isfile(mskfilename):
                sys.exit('ERROR: '+mydefname+'File not found: '+mskfilename)
            with np.load(mskfilename, allow_pickle=True) as mskfile_loaded:
                promask = mskfile_loaded['promask']
                class_ids = [] #mskfile_loaded['classids']
                if self.keepinmemory:
                    self.prepared_masks[image_id] = promask
                    self.prepared_classids[image_id] = class_ids
        else:
            promask = self.prepared_masks[image_id]
            class_ids = self.prepared_classids[image_id]
            # release resources
            self.prepared_masks[image_id] = None
            self.prepared_classids[image_id] = None

#        efflen = len(promask)
#        with np.printoptions(precision=3, suppress=True):
#            print(name, "index: ", self.get_image_index(name), efflen, efflen//2)
#            print(promask[0,1],class_ids)
#            print(promask[1,0])
#            print(promask[10,10])
#            print(promask[0,efflen//2])
#            print(promask[efflen-1,efflen-2])

        return promask, class_ids



    def alignSeqStr(self, seq, strseq):
        """Align a sequence with a structure.
        seq, the first sequence; strseq, the second, or structure, sequence.
        Return alignment object.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1
        aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
        alnm1 = aligner.align(seq, strseq)[0]
        return alnm1


    def transposeFlattenedArray(self, array, lastexcluded=True):
        """Transpose square matrix flattened to an array and return matrix
        transpose in the same flattened format.
        lastexcluded, if set, the last element is excluded from the transpose
        operation and added to the end of the resulting array.
        """
        length = len(array)
        if length < 1: return 1, array

        last = []
        rank = length
        if lastexcluded:
            last = [array[-1]]
            rank = length-1
        rank1 = math.sqrt(rank)
        rank = int(rank1)
        if rank1 != rank:
            sys.stderr.write('ERROR: transposeFlattenedArray: Invalid matrix dimensions.\n')
            return 0, array


        if lastexcluded:
            arrmtx = array[:-1].reshape(rank, rank)
        else:
            arrmtx = array[:].reshape(rank, rank)
        arrmtxT = np.transpose(arrmtx)
        return 1, np.concatenate((arrmtxT.flatten(), last));


    def preparePromage(self, image_id):
        """Prepare a profile image to be ready for processing
        """
        mydefname = 'preparePromage: '
        info = self.image_info[image_id]
        name = info['name']
        profile_pathname = os.path.join(self.pathpro, name+self.proext)
        covfile_pathname = os.path.join(self.pathcov, name+self.covext)
        pdbfile_pathname = os.path.join(self.pathpdb, name+self.pdbext)

        if not os.path.isfile(profile_pathname):
            sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)
        if not os.path.isfile(covfile_pathname):
            sys.exit('ERROR: '+mydefname+'Xcov file not found: '+covfile_pathname)
        if not os.path.isfile(pdbfile_pathname):
            sys.exit('ERROR: '+mydefname+'PDB file not found: '+pdbfile_pathname)

        prodct = readProfile(profile_pathname)
        proseq = ''.join(prodct['seq'])
#        print(proseq)
#        with np.printoptions(precision=3, suppress=True):
#            [print(x) for x in prodct['pro']]
        procov = readXCovfile(covfile_pathname)
        if len(procov) != len(prodct['pro']):
            sys.exit('ERROR: Inconsistent profile and xcov data.')
#        with np.printoptions(precision=3, suppress=True):
#            for i in range(len(procov)):
#                for j in range(i,len(procov)):
#                    print(i,j,procov[i,j])

        # get pdb sequence of the first chain including HETATM records;
        # th result is a dictionary with the pdbstr['seq'] and pdbstr['num']
        # fields
        pdbstr = getChainSequence(pdbfile_pathname, nohetatm=False)
#        print(pdbstr['seq'])
#        print(pdbstr['num'])

        # get the distances between the residues of the pdb chain;
        # (dictionary of ) distances in a range of [0, 64] are recorded as 
        # well as set of residues involved
        pdbdst, pdbres = getDistances(pdbfile_pathname, 0, 64, hetatm=True)
        # get the list of observed distances and associate them with class ids
        pdbdstlst = self.getSetSrtRndDistances([*pdbdst.values()]);
        # get the array of class ids for given observed distances
        ##class_ids = np.array([self.class_names.index(str(int(d))) for d in pdbdstlst])
        # class ids arranged sequentially
        class_ids = np.array([self.class_names.index(str(int(d))) \
            for d in range(self.config.FST_CLASS,self.config.LST_CLASS+1)])
            ##for d in (0, *range(self.config.FST_CLASS,self.config.LST_CLASS+1))])
#        for ndx in range(len(pdbstr['num'])):
#            nres = pdbstr['num'][ndx]
#            for ndx2 in range(ndx+1,len(pdbstr['num'])):
#                nres2 = pdbstr['num'][ndx2]
#                if (nres,nres2) in pdbdst:
#                    print(nres,nres2,pdbdst[(nres,nres2)])

        alnm = self.alignSeqStr(proseq, pdbstr['seq'])
        alnm1splt = re.split(r'\s+', str(alnm))
        proseq_aln = alnm1splt[0].rstrip()
        pdbseq_aln = alnm1splt[2].rstrip()
#        print(proseq_aln, len(proseq_aln))
#        print(pdbseq_aln, len(pdbseq_aln))
#        print(alnm.aligned)

        if len(proseq_aln) != len(pdbseq_aln):
            sys.exit('ERROR: {}Lengths of aligned sequence and structure differ ({}).'.
                format(mydefname,name))

        # get the dimensions first
        prondx = -1
        pdbresndx = -1
        efflen = 0
        for i in range(len(proseq_aln)):
            if proseq_aln[i] != '-': prondx += 1
            if pdbseq_aln[i] != '-': pdbresndx += 1
            if proseq_aln[i] in ('-','X','x') or pdbseq_aln[i] == '-':
                # do not consider positions where profile has Xs;
                # yet, Xs in the structure sequence may represent non-standard
                # amino acids (e.g., selenomethionine)
                continue
            if not pdbstr['num'][pdbresndx] in pdbres:
                continue
            efflen += 1
            ##print(proseq_aln[i],pdbseq_aln[i],pdbstr['num'][pdbresndx])

        # allocate memory
        nclasses = self.config.NUM_CLASSES
        # NOTE: for considerably reducing memory requirements (actually, 
        # making it feasible), each class is assigned to exactly one instance!
        ##ninstances = len(pdbdstlst)
        ninstances = nclasses
        maxdst = nclasses
        ncovattrs = getNXcovfileAttrs()
        nproattrs = getNProfileAttrs()
        numattrs = self.config.IMAGE_CHANNEL_COUNT
        promage = np.empty([efflen, efflen, numattrs], dtype=np.float32, order='C')
        promask = np.zeros([efflen, efflen, ninstances], dtype=np.bool)
        # initialize promask with background class
        promask[:,:,:]=np.concatenate(([True],np.repeat(False,ninstances-1)));

        # produce promage and its masks
        prondxi = prondxj = -1
        pdbresndxi = pdbresndxj = -1
        effndxi = effndxj = 0
        for i in range(len(proseq_aln)):
            if proseq_aln[i] != '-': 
                prondxj = prondxi
                prondxi += 1
            if pdbseq_aln[i] != '-':
                pdbresndxj = pdbresndxi
                pdbresndxi += 1
            if proseq_aln[i] in ('-','X','x') or pdbseq_aln[i] == '-':
                continue
            resnumi = pdbstr['num'][pdbresndxi]
            if not resnumi in pdbres:
                continue
            for j in range(i,len(proseq_aln)):
                if proseq_aln[j] != '-': prondxj += 1
                if pdbseq_aln[j] != '-': pdbresndxj += 1
                if proseq_aln[j] in ('-','X','x') or pdbseq_aln[j] == '-':
                    continue
                resnumj = pdbstr['num'][pdbresndxj]
                if not resnumj in pdbres:
                    continue
                dst = -1
                if resnumi == resnumj:
                    dst = self.config.FST_CLASS
                elif (resnumi,resnumj) in pdbdst:
                    dst = pdbdst[(resnumi,resnumj)]
                    if dst < 2: dst = self.config.FST_CLASS
                    if dst > maxdst:
                        sys.exit(
                         'ERROR: {}Distance between residues ({},{}) is {} (>64) ({}).'.
                            format(mydefname,resnumi,resnumj,dst,name))
                if dst >= 0:
                    dstcls = self.getDstClass(dst)
                    try:
                        #mskndx = pdbdstlst.index(dstcls)
                        mskndx = self.class_names.index(str(int(dstcls)))
                    except ValueError:
                        sys.exit("ERROR: {}Distance {} between ({},{}) not found in their list ({})".
                            format(mydefname,dst,resnumi,resnumj,name))
                    # first, unset the background class
                    promask[effndxi,effndxj,0] = 0
                    promask[effndxi,effndxj,mskndx] = 1
                promage[effndxi,effndxj,:] = np.concatenate((procov[prondxi,prondxj], prodct['pro'][prondxi], prodct['pro'][prondxj]))
                # make matrices symmetric
                if effndxi < effndxj:
                    if dst >= 0:
                        # unset the background class first
                        promask[effndxj,effndxi,0] = 0
                        promask[effndxj,effndxi,mskndx] = 1
                    # NOTE: profile attributes are exchanged to enforce a machine to average 
                    # over the upper and lower triangles of input matrices
                    ##ret, xcovT = self.transposeFlattenedArray(procov[prondxi,prondxj])
                    ret, xcovT = (1, procov[prondxi,prondxj]) ##no transpose
                    if not ret:
                        sys.exit("ERROR: {}Failed to transpose xcov matrix for pair ({},{}) ({})".
                            format(mydefname,resnumi,resnumj,name))
                    ##promage[effndxj,effndxi,:] = np.concatenate((xcovT, prodct['pro'][prondxj], prodct['pro'][prondxi]))
                    promage[effndxj,effndxi,:] = np.concatenate((xcovT, prodct['pro'][prondxi], prodct['pro'][prondxj])) ##no transpose
                effndxj += 1
            effndxi += 1
            effndxj = effndxi

        if self.prepared_promages[image_id] is not None or \
           self.prepared_masks[image_id] is not None or \
           self.prepared_classids[image_id] is not None:
            sys.exit("ERROR: {}Data buffers at {} are not empty ({}). Terminating..".
                            format(mydefname,image_id,name))

        self.prepared_promages[image_id] = promage
        self.prepared_masks[image_id] = promask
        self.prepared_classids[image_id] = class_ids



    def preparePromageWithoutMask(self, image_id):
        """Prepare a profile image to be ready for processing.
        Mask in not created.
        """
        mydefname = 'preparePromageWithoutMask: '
        info = self.image_info[image_id]
        name = info['name']
        profile_pathname = os.path.join(self.pathpro, name+self.proext)
        covfile_pathname = os.path.join(self.pathcov, name+self.covext)

        if not os.path.isfile(profile_pathname):
            sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)
        if not os.path.isfile(covfile_pathname):
            sys.exit('ERROR: '+mydefname+'Xcov file not found: '+covfile_pathname)

        prodct = readProfile(profile_pathname)
        proseq = ''.join(prodct['seq'])
#        print(proseq)
#        with np.printoptions(precision=3, suppress=True):
#            [print(x) for x in prodct['pro']]
        procov = readXCovfile(covfile_pathname)
        if len(procov) != len(prodct['pro']):
            sys.exit('ERROR: Inconsistent profile and xcov data.')
#        with np.printoptions(precision=3, suppress=True):
#            for i in range(len(procov)):
#                for j in range(i,len(procov)):
#                    print(i,j,procov[i,j])

        # get the dimensions first
        efflen = len(proseq)
        # allocate memory
        ncovattrs = getNXcovfileAttrs()
        nproattrs = getNProfileAttrs()
        numattrs = self.config.IMAGE_CHANNEL_COUNT
        promage = np.empty([efflen, efflen, numattrs], dtype=np.float32, order='C')

        # produce promage and its masks
        prondxi = prondxj = -1
        pdbresndxi = pdbresndxj = -1
        effndxi = effndxj = 0
        for i in range(len(proseq)):
            prondxj = prondxi
            prondxi += 1
            ##if proseq[i] in ('-','X','x'):
            ##    continue
            for j in range(i,len(proseq)):
                prondxj += 1
                ##if proseq[j] in ('-','X','x'):
                ##    continue
                promage[effndxi,effndxj,:] = np.concatenate((procov[prondxi,prondxj], prodct['pro'][prondxi], prodct['pro'][prondxj]))
                # make matrices symmetric
                if effndxi < effndxj:
                    # NOTE: profile attributes are exchanged to enforce a machine to average 
                    # over the upper and lower triangles of input matrices
                    ##ret, xcovT = self.transposeFlattenedArray(procov[prondxi,prondxj])
                    ret, xcovT = (1, procov[prondxi,prondxj]) ##no transpose
                    if not ret:
                        sys.exit("ERROR: {}Failed to transpose xcov matrix for pair ({},{}) ({})".
                            format(mydefname,prondxi,prondxj,name))
                    ##promage[effndxj,effndxi,:] = np.concatenate((xcovT, prodct['pro'][prondxj], prodct['pro'][prondxi]))
                    promage[effndxj,effndxi,:] = np.concatenate((xcovT, prodct['pro'][prondxi], prodct['pro'][prondxj])) ##no transpose
                effndxj += 1
            effndxi += 1
            effndxj = effndxi

        if self.prepared_promages[image_id] is not None:
            sys.exit("ERROR: {}Data buffers at {} are not empty ({}). Terminating..".
                            format(mydefname,image_id,name))

        self.prepared_promages[image_id] = promage


    def prepareMask(self, image_id, pdbfile_pathname, length=0):
        """Prepare a mask from a pdb file. The (expected) length of the 
        structure is given by length.
        """
        mydefname = 'prepareMask: '
        name = os.path.basename(pdbfile_pathname)

        if not os.path.isfile(pdbfile_pathname):
            sys.exit('ERROR: '+mydefname+'PDB file not found: '+pdbfile_pathname)

        # get pdb sequence of the first chain excluding HETATM records;
        # th result is a dictionary with the pdbstr['seq'] and pdbstr['num']
        # fields
        pdbstr = getChainSequence(pdbfile_pathname, nohetatm=True)
#        print(pdbstr['seq'])
#        print(pdbstr['num'])

        # get the distances between the residues of the pdb chain;
        # (dictionary of ) distances in a range of [0, 64] are recorded as 
        # well as set of residues involved
        pdbdst, pdbres = getDistances(pdbfile_pathname, 0, 64, hetatm=False)
        # get the list of observed distances and associate them with class ids
        pdbdstlst = self.getSetSrtRndDistances([*pdbdst.values()]);
        # get the array of class ids for given observed distances
        ##class_ids = np.array([self.class_names.index(str(int(d))) for d in pdbdstlst])
        # class ids arranged sequentially
        class_ids = np.array([self.class_names.index(str(int(d))) \
            for d in range(self.config.FST_CLASS,self.config.LST_CLASS+1)])
            ##for d in (0, *range(self.config.FST_CLASS,self.config.LST_CLASS+1))])
#        for ndx in range(len(pdbstr['num'])):
#            nres = pdbstr['num'][ndx]
#            for ndx2 in range(ndx+1,len(pdbstr['num'])):
#                nres2 = pdbstr['num'][ndx2]
#                if (nres,nres2) in pdbdst:
#                    print(nres,nres2,pdbdst[(nres,nres2)])

        ## length is given
        if 0 < length:
            efflen = length
        else:
            # get the dimensions first
            pdbresndx = -1
            efflen = 0
            for i in range(len(pdbstr['seq'])):
                pdbresndx += 1
                if not pdbstr['num'][pdbresndx] in pdbres:
                    continue
                efflen = pdbstr['num'][pdbresndx]
                ##print(proseq_aln[i],pdbseq_aln[i],pdbstr['num'][pdbresndx])

        # allocate memory
        nclasses = self.config.NUM_CLASSES
        # NOTE: for considerably reducing memory requirements (actually, 
        # making it feasible), each class is assigned to exactly one instance!
        ##ninstances = len(pdbdstlst)
        ninstances = nclasses
        maxdst = nclasses
        promask = np.zeros([efflen, efflen, ninstances], dtype=np.bool)
        # initialize promask with background class
        promask[:,:,:]=np.concatenate(([True],np.repeat(False,ninstances-1)));

        # produce the mask
        pdbresndxi = pdbresndxj = -1
        for i in range(len(pdbstr['seq'])):
            pdbresndxj = pdbresndxi
            pdbresndxi += 1
            resnumi = int(pdbstr['num'][pdbresndxi])
            if not resnumi in pdbres:
                continue
            if resnumi < 1 or efflen < resnumi:
                sys.exit('ERROR: {}Residue out of bounds: %d'%(mydefname,resnumi))
            for j in range(i,len(pdbstr['seq'])):
                pdbresndxj += 1
                resnumj = int(pdbstr['num'][pdbresndxj])
                if not resnumj in pdbres:
                    continue
                if resnumj < resnumi or efflen < resnumj:
                    sys.exit('ERROR: %sInvalid residue or out of bounds: %d (%s): Condition (%d<%d) or (%d<%d)'%(
                        mydefname,resnumj,name, resnumj,resnumi, efflen,resnumj))
                dst = -1
                if resnumi == resnumj:
                    dst = self.config.FST_CLASS
                elif (resnumi,resnumj) in pdbdst:
                    dst = pdbdst[(resnumi,resnumj)]
                    if dst < 2: dst = self.config.FST_CLASS
                    if dst > maxdst:
                        sys.exit(
                         'ERROR: {}Distance between residues ({},{}) is {} (>64) ({}).'.
                            format(mydefname,resnumi,resnumj,dst,name))
                if dst >= 0:
                    dstcls = self.getDstClass(dst)
                    try:
                        #mskndx = pdbdstlst.index(dstcls)
                        mskndx = self.class_names.index(str(int(dstcls)))
                    except ValueError:
                        sys.exit("ERROR: {}Distance {} between ({},{}) not found in their list ({})".
                            format(mydefname,dst,resnumi,resnumj,name))
                    # first, unset the background class
                    promask[resnumi-1,resnumj-1,0] = 0
                    promask[resnumi-1,resnumj-1,mskndx] = 1
                # make matrices symmetric
                if resnumi < resnumj:
                    if dst >= 0:
                        # unset the background class first
                        promask[resnumj-1,resnumi-1,0] = 0
                        promask[resnumj-1,resnumi-1,mskndx] = 1

        if self.prepared_masks[image_id] is not None or \
           self.prepared_classids[image_id] is not None:
            sys.exit("ERROR: {}Data buffers at {} are not empty ({}). Terminating..".
                            format(mydefname,image_id,name))

        self.prepared_masks[image_id] = promask
        self.prepared_classids[image_id] = class_ids



## ===========================================================================
##
if __name__ == '__main__':
    ## test routines
    basedir = '/data/CASP14_datasets/'
    # directory to save logs and trained model
    MODEL_DIR = os.path.join(basedir, 'mrcnn_model1')

    ret, options = ParseArguments()
    if not ret: sys.exit()

    config = PromageConfig()

    if not (options.make1pathname or options.mask1pathname or 
            options.make or options.train or 
            options.visual or options.pmginfer or options.stats):
        sys.stderr.write("\nInvoke the program with -h to see available options.\n")
        sys.exit()


    if options.make1pathname:
        promage = PromageDataset(config)
        promage.makePromage1(options.make1pathname)
        if options.makemasks:
            promage.makeMasksOnly(options.makemasks)
        sys.exit()


    if options.mask1pathname:
        promage = PromageDataset(config)
        promage.makeMask1(options.mask1pathname)
        sys.exit()


    if options.visual:
        # Device to load the neural network on.
        DEVICE = "/cpu:0"
        # Create model in inference mode
        with tf.device(DEVICE):
            model = mrclib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, 
                                    config=config)
        model_path = model.find_last()
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        # Show stats of all trainable weights    
        table = visualize.display_weight_stats(model, display=0)
        tbl_fmt = ""
        for row in table:
            for col in row:
                tbl_fmt += ' {:40} '.format(str(col))
            tbl_fmt += '\n'
        print(tbl_fmt)
        sys.exit()



    if options.pmginfer:
        infconfig = InferencePromageConfig()
        promage_infer = PromageDataset(infconfig)
        promage_infer.loadPromage1(options.pmginfer)
        # load the model in inference mode
        model = mrclib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, 
                                config=infconfig)
        # path to saved weights
        model_path = model.find_last()
        # load trained weights
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        # inference
        results = model.detect([promage_infer.load_image(0)], verbose=1)
        r1 = results[0]
        # results
        print(r1['class_ids'])
        print(r1['scores'])
        print(r1['masks'])
        print(np.nonzero(r1['masks']))
        sys.exit()



    trnset_prodir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_pro__training'
    trnset_covdir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_cov__training'
    trnset_pdbdir = basedir + 'pdb70_from_mmcif_200205__selection__pdb__training'
    ##optional directory for prepared promages:
    trnset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__SEMSEGM'
    ##trnset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__training'

    promages_train = PromageDataset(config, train=True)
    promages_train.loadPromages(prodir=trnset_prodir, covdir=trnset_covdir, 
            pdbdir=trnset_pdbdir, outdir=trnset_pmgdir, make=options.make)
#    promages_train.load_image(994)
#    promages_train.load_mask(994)



    valset_prodir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_pro__validation'
    valset_covdir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_cov__validation'
    valset_pdbdir = basedir + 'pdb70_from_mmcif_200205__selection__pdb__validation'
    ##optional directory for prepared promages:
    valset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__SEMSEGM'
    ##valset_pmgdir = basedir + 'pdb70_from_mmcif_200205__selection__promage__validation'

    promages_valid = PromageDataset(config, val=True)
    promages_valid.loadPromages(prodir=valset_prodir, covdir=valset_covdir,
            pdbdir=valset_pdbdir, outdir=valset_pmgdir, make=options.make)
#    promages_valid.load_image(0)
#    promages_valid.load_mask(0)


    if options.stats:
        _ = promages_train.class_stats(doprint=True)
        sys.exit()


    if options.make:
        sys.exit()


    if options.train:
        config.LEARNING_RATE = options.lrate

        # model in training mode
        model = mrclib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
        # load the last model trained and continue training
        try:
            if not options.restart:
                model_path = model.find_last()
                print("Loading weights from ", model_path)
                model.load_weights(model_path, by_name=True)
        except FileNotFoundError:
            print("Saved weights not found. Initializing.")

        # train the network;
        # it is possible to pass a regular expression to select
        # which layers to train by name pattern.
        model.train(promages_train, promages_valid, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='all')

#<<>>
