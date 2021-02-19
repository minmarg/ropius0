#!/usr/bin/env python
##(C)2019-2021 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University
import os,time,cv2, sys, math
import fnmatch
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
from joblib import Parallel, delayed
import time

sys.path.insert(1, '/data/installed-software/ROPIUS0/bin')
sys.path.insert(1, '/data/installed-software/Semantic-Segmentation-Suite')
from utils import utils, helpers
from builders import model_builder

# private imports
from mrcnn.utils import resize_image, resize_mask
import promage4segm_519 as pmg

# initialize seeds to obtain consistently the same result
np.random.seed(2020)
tf.set_random_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument('--promagedir', type=str, default=None, required=True, help='Directory of promages to make predictions on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=128, help='Height of cropped input image to network.')
parser.add_argument('--crop_width', type=int, default=128, help='Width of cropped input image to network.')
parser.add_argument('--batchsize', type=int, default=1, help='Promage batch size (<=32).')
parser.add_argument('--ncpus', type=int, default=1, help='Max number of CPUs to use.')
parser.add_argument('--model', type=str, default='Encoder-Decoder-Skip', required=False, help='The model being used.')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend being used.')
parser.add_argument('--dataset', type=str, default="SEMSEGM", required=False, help='The dataset in use.')
parser.add_argument('--output', type=str, required=False, help='Output directory of predictions.')
parser.add_argument('--mean', action='store_true', help='Calculate the mean predicted distance for each pair of residues.')
parser.add_argument('--printall', action='store_true', help='Save all predicted probabilities instead of binned.')
args = parser.parse_args()


class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

if 32 < args.batchsize or args.batchsize < 1:
    sys.exit("ERROR: Invalid batch size specified: %d"%(args.batchsize))

if multiprocessing.cpu_count() < args.ncpus or args.ncpus < 1:
    sys.exit("ERROR: Invalid number of CPUs specified: %d"%(args.ncpus))

if not os.path.exists(args.promagedir):
    sys.exit("ERROR: Input directory does not exist: %s"%(args.promagedir))

if os.path.isfile(args.promagedir):
    sys.exit("ERROR: Specified input is a file: %s"%(args.promagedir))

if not args.output:
    sys.exit("ERROR: Output directory is not provided")
elif not os.path.exists(args.output):
    os.mkdir(args.output)
elif os.path.isfile(args.output):
    sys.exit("ERROR: Output directory is a file: %s"%(args.output))



print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Promage dir -->", args.promagedir)
print("Mean -->", args.mean)

pmgext = '.pmg.npz' ## file extension of compressed preprocessed promages

# Initializing network
##{{MM
config = tf.ConfigProto(device_count={'GPU':0},gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(0)),log_device_placement=False)
##}}
##config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

##{{MM
conf = pmg.PromageConfig()
##step in x and y directions for predicting the whole promage;
##NOTE: values <= 0 indicate using no manual incremental prediction
MEANDIST = args.mean ##calculate mean distance instead of a one with max prob
SUBT_MEAN = False
NORMALIZE = True #False
EEF_NUM_CLASSES = conf.NUM_CLASSES ##effective number of classes
net_input = tf.placeholder(tf.float32,shape=[None,None,None,conf.IMAGE_CHANNEL_COUNT])
##}}
##net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

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
    #if SUBT_MEAN: image = image.astype(np.float32) - conf.MEAN_PIXEL
    #if NORMALIZE: image = image / conf.PIXEL_NORM
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



def getPromagelist(directory, ext):
    """Read `directory' for promages with extension `ext', sort them by 
    size and return the resulting list.
    """
    pmgfiles = []

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and fnmatch.fnmatch(entry, '*' + ext):
                pmgfiles.append(entry.name.rsplit(ext)[0]) ##add filename only

    for i in range(len(pmgfiles)):
        pmgfiles[i] = (pmgfiles[i], os.path.getsize(os.path.join(directory,pmgfiles[i]+ext)))

    #sort by file size
    pmgfiles.sort(key=lambda pmgname: pmgname[1], reverse=True)

    for i in range(len(pmgfiles)):
        pmgfiles[i] = pmgfiles[i][0]

    return pmgfiles



def readData(i, j, step, pmgnamelist):
    """Collect a batch of images.
    """
    index = i * step + j
    if len(pmgnamelist) <= index: return
    pmgname = pmgnamelist[index]

    with tf.device('/cpu:0'):
        promage_infer = pmg.PromageDataset(conf, keepinmemory=True)
        promage_infer.loadPromage1(os.path.join(args.promagedir, pmgname))
        loaded_promage = promage_infer.load_image(0)
        if SUBT_MEAN: loaded_promage = loaded_promage.astype(np.float32) - conf.MEAN_PIXEL
        if NORMALIZE: loaded_promage = loaded_promage / conf.PIXEL_NORM

    return pmgname, loaded_promage.shape, promage_infer, loaded_promage



def processOutput(output_probs, promage_window, promage_name, promagedinfo):
    """Process prediction obtained for one promage. Calculate appropriately 
    probabilities, write them to file. Also, write a graphical 
    representation of the prediction to file.
    """
    output_probs = output_probs[promage_window[0]:promage_window[2],promage_window[1]:promage_window[3]]
    output_mean = np.zeros(output_probs.shape[:2])
    if MEANDIST:
        patch_mean = np.dot( output_probs[:,:,1:], np.arange(1,num_classes) )
        patch_mean_prbs = np.squeeze(
            np.take_along_axis(output_probs, 
                np.around(patch_mean[:,:,np.newaxis]).astype(int), 
                axis=-1), 
            axis=-1);
        patch_mean[patch_mean_prbs<output_probs[:,:,0]] = 0
        output_mean = np.around(patch_mean)
    else:
        output_mean[:,:] = np.argmax(output_probs[:,:,:], axis=-1)
    ##output_mean_probs = np.squeeze(np.take_along_axis(output_probs, output_mean[:,:,np.newaxis].astype(int), axis=-1), axis=-1);
    ##print(output_mean)

    if EEF_NUM_CLASSES < conf.NUM_CLASSES:
        output_mean[output_mean>EEF_NUM_CLASSES-1] = 0 ##<<<


    filename = os.path.join(args.output, promage_name)
    predfilename = '%s_predmean.png'%(filename) if MEANDIST else '%s_pred.png'%(filename)
    probfilename = '%s_predmean.prb'%(filename) if MEANDIST else '%s_pred.prb'%(filename)

    avg2_probs = np.zeros(num_classes)


    with open(probfilename,'w') as fp:
        for i in range(output_mean.shape[0]):
            for j in range(i+1,output_mean.shape[0]):
                avg2_probs = (output_probs[i,j,:]+output_probs[j,i,:])*0.5
                if MEANDIST:
                    dstname = int(round( np.dot( avg2_probs[1:], np.take(promagedinfo.class_names,np.arange(1,num_classes)).astype(int) ) ))
                    clsndx = promagedinfo.class_names.index(str(dstname)) if 0 < dstname else 0
                else:
                    clsndx = np.argmax(avg2_probs)
                    dstname = int(promagedinfo.class_names[clsndx]) if 0 < clsndx else 0
                clsprb = avg2_probs[clsndx]
                if clsndx == 0 or clsprb < avg2_probs[0]:
                    ##either background or background probability is greater than the mean
                    continue

                ##clsndx = int(round((output_mean[i,j] + output_mean[j,i]) * 0.5))
                ##if clsndx == 0:
                ##    continue
                ##clsprb = (output_mean_probs[i,j] + output_mean_probs[j,i]) * 0.5
                ##dstname = int(promagedinfo.class_names[clsndx])

                fp.write('%d %d  %.1f %.3f   '%(i+1, j+1, dstname, clsprb))
                if args.printall:
                    [fp.write(' %.3f'%(avg2_probs[ip])) for ip in range(1,num_classes)]
                    fp.write('  %.3f'%(avg2_probs[0])) ##background
                else:
                    fp.write(' %.3f'%(sum(avg2_probs[1:4]))) ##dst<=4A
                    [fp.write(' %.3f'%(sum(avg2_probs[ip:ip+2]))) for ip in range(4,20,2)] ##4A..20A in steps of 2A
                    fp.write(' %.3f'%(sum(avg2_probs[[0,*range(20,num_classes)]]))) ##dst>20A
                fp.write('\n')
                ##TEST:
                ##[fp.write(' %.3f'%((output_probs[i,j,ip]+output_probs[j,i,ip])*0.5)) for ip in range(num_classes)]
                ###[fp.write(' %.3f'%(output_probs[i,j,ip])) for ip in range(num_classes)]
                ###fp.write('\n')
                ###[fp.write(' %.3f'%(output_probs[j,i,ip])) for ip in range(num_classes)]
                ##fp.write('\n')


    out_vis_image = helpers.colour_code_segmentation(output_mean, label_values)
    cv2.imwrite(predfilename, cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))



print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


print("Reading directory " + args.promagedir)
pmgfiles = getPromagelist(args.promagedir, pmgext)

input_promage_batch = []

num_iters = int(np.ceil(len(pmgfiles) / args.batchsize))

ncpus = min(args.ncpus, args.batchsize)

st = time.time()

print("Starting prediction...")

for i in range(num_iters):
    
    bsize = args.batchsize
    if len(pmgfiles) < (i+1) * args.batchsize: bsize = len(pmgfiles) - i * args.batchsize

    ##NOTE: it depends on a python verion, but some do not allow reading bytes more than max_int in parallel
    inputs = Parallel(n_jobs=ncpus)(delayed(readData)(i, j, args.batchsize, pmgfiles) for j in range(bsize))
    promage_names, promage_shapes, promage_dinfos, inputpromages = zip(*inputs)

    ndxmaxshape = promage_shapes.index(max(promage_shapes, key=lambda shp:shp[0]))

    inputpromages_ndxmaxshape, promage_windows_ndxmaxshape, _ = pad64_image(inputpromages[ndxmaxshape])

    inputs = Parallel(n_jobs=ncpus)(delayed(pad_image)(inputpromages[j], inputpromages_ndxmaxshape.shape[0]) for j in range(bsize))
    inputpromages, promage_wins, _ = zip(*inputs)
    promage_shapes = [inputpromages[p].shape for p in range(len(inputpromages))]

    promage_windows = [promage_wins[j] for j in range(len(promage_wins))]
    promage_windows[ndxmaxshape] = promage_windows_ndxmaxshape


    [input_promage_batch.append(np.expand_dims(inputpromages[p], axis=0)) for p in range(len(inputpromages))]

    if len(input_promage_batch) == 1:
        input_promage_batch = input_promage_batch[0]
    else:
        input_promage_batch = np.squeeze(np.stack(input_promage_batch, axis=1))


    #_, output_probs_batch = sess.run([network,probs],feed_dict={net_input:input_image_batch})
    output_probs_batch = sess.run(probs,feed_dict={net_input:input_promage_batch})

    Parallel(n_jobs=ncpus)(
        delayed(processOutput)(output_probs_batch[j], promage_windows[j], promage_names[j], promage_dinfos[j]) 
            for j in range(bsize))

    print("  batch of %d promages processed"%(len(output_probs_batch)))

    input_promage_batch = []


print('\nFinished (time elapsed: %.2f).\n'%(time.time()-st))

