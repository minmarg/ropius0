#!/usr/bin/env python
## (C)2020 Mindaugas Margelevicius, Vilnius University
## some parts of the code adapted from Semantic Segmentation Suite
import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'bin'))
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'infer'))
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'Semantic-Segmentation-Suite'))

from utils import utils, helpers
from builders import model_builder

# private imports
from mrcnn.utils import resize_image, resize_mask
import promage4segm_519 as pmg

# initialize seeds to obtain consistently the same result
np.random.seed(2020)
tf.set_random_seed(7)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=128, help='Height of cropped input image to network.')
parser.add_argument('--crop_width', type=int, default=128, help='Width of cropped input image to network.')
parser.add_argument('--model', type=str, default='Encoder-Decoder-Skip', required=False, help='The model being used.')
parser.add_argument('--frontend', type=str, default="", help='The frontend being used.') ##ResNet101
parser.add_argument('--dataset', type=str, default="SEMSEGM", required=False, help='The dataset in use.')
parser.add_argument('--filter', action='store_true', required=False, help='Filter out samples of too small size.')
parser.add_argument('--out', type=str, required=True, help='Output filename.')
parser.add_argument('--dst', type=int, default=20, help='Alternative upper distance threshold.')
parser.add_argument('--gpu', type=int, default=0, help='ID of GPU to use.')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin evaluation *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)

# Initializing network
##{{MM
config = tf.ConfigProto(device_count={'GPU':1},gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(args.gpu)),log_device_placement=False)
##}}
##config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Load the data
print("Loading the validation data...")
_,_, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

##{{MM
conf = pmg.PromageConfig()
MEANDIST = False ##calculate mean distance instead of a one with max prob
SUBT_MEAN = False
NORMALIZE = True #False
EEF_NUM_CLASSES = conf.NUM_CLASSES ##effective number of classes
net_input = tf.placeholder(tf.float32,shape=[None,None,None,conf.IMAGE_CHANNEL_COUNT])
##}}
##net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

## load validation promages
basedir = '/data/CASP14_datasets/'
valset_prodir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_pro__validation'
valset_covdir = basedir + 'pdb70_from_mmcif_200205__selection__comer2_cov__validation'
valset_pdbdir = basedir + 'pdb70_from_mmcif_200205__selection__pdb__validation'
promages_val = pmg.PromageDataset(conf, val=True, keepinmemory=False)
promages_val.loadPromages(prodir=valset_prodir, covdir=valset_covdir,
        pdbdir=valset_pdbdir, outdir=args.dataset)

if args.filter:
    # filter out samples of small size
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

network, _ = model_builder.build_model(args.model, frontend=args.frontend,
                                        net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)
##{{MM
probs = tf.nn.softmax(logits=network)
##}}

sess.run(tf.global_variables_initializer())



def pad_image(image, min_dim=None):
    """Pad image with zeros
    """
    # return results in the same dtype
    image_dtype = image.dtype
    if min_dim is None:# or min_dim % 64 != 0:
        raise Exception("Dimension is not multiple of 64 in mode {}".format(mode))
    scale = 1
    h, w = image.shape[:2]
    # Height
    if h < min_dim:
        max_h = min_dim
        ## the origin is better for prediction:
        top_pad = 0 #(max_h - h) // 2
        bottom_pad = max_h - h - top_pad
    else:
        top_pad = bottom_pad = 0
    # Width
    if w < min_dim:
        max_w = min_dim
        ## the origin is better for prediction:
        left_pad = 0 #(max_w - w) // 2
        right_pad = max_w - w - left_pad
    else:
        left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, padding



def pad64_image(image):
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
    return image.astype(image_dtype), window, padding



def readData(i, j, step):
    """Collect a batch of images.
    """
    index = i * step + j
    id = id_list[index]
    ##{{MM
    name = os.path.basename(val_input_names[id]).rsplit('.', 2)[0]
    name_index = promages_val.get_image_index(name)
    input_image = promages_val.load_image(name_index)
    gt, _ = promages_val.load_mask(name_index)
    image_shape = input_image.shape
    ##}}

    with tf.device('/cpu:0'):
        ##{{MM
        input_image, window, _ = pad64_image(input_image)
        ##}}

        # Prep the data. Make sure the labels are in one-hot format
        ##{{MM
        if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
        if NORMALIZE: input_image = input_image / conf.PIXEL_NORM
        ##}}

        lfunc = lambda ndx: int(promages_val.class_names[ndx])
        ndx2dfunc = np.vectorize(lfunc)
        ##{{MM:mask os already one-hot-coded}}
        ##gt = np.float32(helpers.one_hot_it(label=gt, label_values=label_values))
        gt = helpers.reverse_one_hot(gt)
        gt_dstname = np.full(gt.shape, conf.LST_CLASS + 1)
        gt_dstname[gt>0] = ndx2dfunc(gt[gt>0])

    return name, image_shape, window, input_image, gt_dstname





print('Loading model checkpoint weights')
print('(%s)'%(args.checkpoint_path))
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

maxnparallelworkers = 10

# equivalent to shuffling
id_list = np.random.permutation(len(val_input_names))

num_iters = int((len(id_list) + maxnparallelworkers - 1) / maxnparallelworkers)

dst8 = 8
cnt = 0

n_vrnt = 3
mae_mean = np.zeros(n_vrnt); std_mean = np.zeros(n_vrnt)
mae_dst_mean = np.zeros(n_vrnt); std_dst_mean = np.zeros(n_vrnt)
mae_dst8_mean = np.zeros(n_vrnt); std_dst8_mean = np.zeros(n_vrnt)
mae_dst8_r_mean= np.zeros(n_vrnt); std_dst8_r_mean = np.zeros(n_vrnt)

timestart = time.time()

fp = open(args.out,'w')

fp.write('\n## 1st line: Full matrix; 2nd: Upper triangle; 3rd: Lower triangle\n##\n\n')

for i in range(num_iters):
    nparallelworkers = len(id_list) - i * maxnparallelworkers
    if maxnparallelworkers < nparallelworkers: nparallelworkers = maxnparallelworkers
    inputoutput = Parallel(n_jobs=nparallelworkers)(delayed(readData)(i, j, nparallelworkers) for j in range(nparallelworkers))
    image_names, image_shapes, image_windows, inputimages, outputimages = zip(*inputoutput)

    for k in range(nparallelworkers):
        input_image_batch = []

        name = image_names[k]

        input_image_batch.append(np.expand_dims(inputimages[k], axis=0))
        input_image_batch = input_image_batch[0]

        ## prediction
        output_probs_batch = sess.run(probs,feed_dict={net_input:input_image_batch})

        output_probs = output_probs_batch[0][image_windows[k][0]:image_windows[k][2],image_windows[k][1]:image_windows[k][3],:]

        gt_dstname = outputimages[k]

        ##output_probs[:,:,:] = 0 ##TEST:
        ##gt_dstname_tst = gt_dstname - 1 ##TEST:
        ##gt_dstname_tst[gt_dstname_tst==conf.LST_CLASS] = 0 ##TEST:
        ##np.put_along_axis(output_probs, gt_dstname_tst[:,:,np.newaxis], 1, axis=-1) ##TEST:

        if MEANDIST:
            lfunc = lambda dval: promages_val.class_names.index(str(dval))
            d2ndxfunc = np.vectorize(lfunc)
            dstname = np.round( np.dot( output_probs[:,:,1:], 
                                        np.take(promage_infer.class_names,np.arange(1,num_classes)).astype(int) )
                              ).astype(int)
            clsndx = np.zeros(dstname.shape)
            clsndx[dstname>0] = d2ndxfunc(dstname[dstname>0])
            clsprb = np.squeeze(
                np.take_along_axis(output_probs, clsndx[:,:,np.newaxis], axis=-1),
                axis=-1);
            dstname[output_probs[:,:,0]>clsprb] = conf.LST_CLASS + 1
            dstname[dstname==0] = conf.LST_CLASS + 1
        else:
            lfunc = lambda ndx: int(promages_val.class_names[ndx])
            ndx2dfunc = np.vectorize(lfunc)
            clsndx = np.argmax(output_probs, axis=-1)
            dstname = np.full(clsndx.shape, conf.LST_CLASS + 1)
            dstname[clsndx>0] = ndx2dfunc(clsndx[clsndx>0])

        ## mask subdiagonals (matching sequence separation) for excluding from consideration
        np.fill_diagonal(dstname,0)
        for sub in range(1,4):
            np.fill_diagonal(dstname[sub:],0)
            np.fill_diagonal(dstname[:,sub:],0)

        for vnt in range(n_vrnt):
            if vnt == 1:
                dstname_vnt = np.triu(dstname)
            elif vnt == 2:
                dstname_vnt = np.tril(dstname)
            else:
                dstname_vnt = np.copy(dstname)

            absv = np.abs(gt_dstname[(dstname_vnt>0) & ((dstname_vnt<=args.dst)|(gt_dstname<=args.dst))] - 
                         dstname_vnt[(dstname_vnt>0) & ((dstname_vnt<=args.dst)|(gt_dstname<=args.dst))])
            mae_dst = np.mean(absv)
            std_dst = np.std(absv)
            n_dst = absv.shape[0]

            absv = np.abs(gt_dstname[(dstname_vnt>0) & (dstname_vnt<=dst8)] - dstname_vnt[(dstname_vnt>0) & (dstname_vnt<=dst8)])
            mae_dst8 = np.mean(absv)
            std_dst8 = np.std(absv)
            n_dst8 = absv.shape[0]

            absv = np.abs(gt_dstname[(dstname_vnt>0) & (gt_dstname<=dst8)] - dstname_vnt[(dstname_vnt>0) & (gt_dstname<=dst8)])
            mae_dst8_r = np.mean(absv)
            std_dst8_r = np.std(absv)
            n_dst8_r = absv.shape[0]

            absv = np.abs(gt_dstname[dstname_vnt>0] - dstname_vnt[dstname_vnt>0])
            mae = np.mean(absv)
            std = np.std(absv)
            n_full = absv.shape[0]

            fp.write(' %3d %10s: mae= %6.3f std= %6.3f shape= %-4d x %-4d n= %8d'
                     ' | dst %2d: mae= %6.3f std= %6.3f n= %7d'
                     ' | dst %2d: mae= %6.3f std= %6.3f n= %7d'
                     ' recall: mae= %6.3f std= %6.3f n= %7d\n'%(
                cnt,name,mae,std,dstname.shape[0],dstname.shape[1],n_full,
                args.dst,mae_dst,std_dst,n_dst,
                dst8,mae_dst8,std_dst8,n_dst8, mae_dst8_r,std_dst8_r,n_dst8_r))

            mae_mean[vnt] += mae; mae_dst_mean[vnt] += mae_dst; mae_dst8_mean[vnt] += mae_dst8; mae_dst8_r_mean[vnt] += mae_dst8_r
            std_mean[vnt] += std; std_dst_mean[vnt] += std_dst; std_dst8_mean[vnt] += std_dst8; std_dst8_r_mean[vnt] += std_dst8_r

        fp.write('\n')
        cnt += 1

    ##break ##TEST:

elapsed = time.time() - timestart

fp.write('\n## 1st line: Full matrix; 2nd: Upper triangle; 3rd: Lower triangle\n##\n\n')

if not cnt: cnt = 1
for vnt in range(n_vrnt):
    fp.write('Mean: MAE= %.3f STD= %.3f | dst %2d: MAE= %.3f STD= %.3f | '
             'dst %2d: MAE= %.3f STD= %.3f recall: MAE= %.3f STD= %.3f\n'%(
        mae_mean[vnt]/cnt, std_mean[vnt]/cnt,
        args.dst, mae_dst_mean[vnt]/cnt, std_dst_mean[vnt]/cnt,
        dst8, mae_dst8_mean[vnt]/cnt, std_dst8_mean[vnt]/cnt, mae_dst8_r_mean[vnt]/cnt, std_dst8_r_mean[vnt]/cnt))

fp.write('\nElapsed: %s\n'%(time.strftime("%H:%M:%S", time.gmtime(elapsed))))

fp.close()

