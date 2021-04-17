#!/usr/bin/env python
import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'bin'))
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, 'Semantic-Segmentation-Suite'))

from utils import utils, helpers
from builders import model_builder

# private imports
from mrcnn.utils import resize_image, resize_mask
import promage4segm_519 as pmg

# set print options
np.set_printoptions(threshold=sys.maxsize)

# initialize seeds to obtain consistently the same result
np.random.seed(2020)
tf.set_random_seed(7)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inpmg', type=str, required=True, help='Input promage (without extension).')
parser.add_argument('--inmskdir', type=str, required=True, help='Input directory of masks for the same promage.')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=128, help='Height of cropped input image to network.')
parser.add_argument('--crop_width', type=int, default=128, help='Width of cropped input image to network.')
parser.add_argument('--model', type=str, default='Encoder-Decoder-Skip', required=False, help='The model being used.')
parser.add_argument('--frontend', type=str, default="", help='The frontend being used.') ##ResNet101
parser.add_argument('--dataset', type=str, default="SEMSEGM", required=False, help='The dataset in use.')
parser.add_argument('--range', type=str, default="full", required=False,
                        help='Range over which to assess the correspondence between the model and given masks. '
                             'Possible values are "full", "upper" (upper triangle), and "lower" (lower triangle).')
parser.add_argument("--target", type=str, required=True, help="Target name.")
parser.add_argument("--modelnum", type=int, required=True, help="Model number.")
parser.add_argument('--out', type=str, required=True, help='Output filename.')
parser.add_argument('--dst', type=int, default=67, help='Upper distance threshold.')
parser.add_argument('--prb', type=float, default=0.0, help='Consider promage distances of at least this probability.')
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
config = tf.ConfigProto(device_count={'GPU':0},gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(0)),log_device_placement=False)
##}}
##config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

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


promage = pmg.PromageDataset(conf)
promage.loadPromage1(args.inpmg)
maskfiles = promage.getFiles(args.inmskdir, remext=False)
masks = []
for i in range(len(maskfiles)): masks.append(pmg.PromageDataset(conf))



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



def readPromage():
    """Read a promage.
    """
    input_image = promage.load_image(0)
    image_shape = input_image.shape

    input_image, window, _ = pad64_image(input_image)

    if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
    if NORMALIZE: input_image = input_image / conf.PIXEL_NORM

    return image_shape, window, input_image



def readMask(i, j, step):
    """Read a mask corresponding to the given promage.
    """
    index = i * step + j
    name = os.path.basename(maskfiles[index]).rsplit('.', 2)[0]
    pathname = os.path.join(args.inmskdir, name)
    ##{{MM
    masks[index].loadPromage1(pathname)
    pmask, _ = masks[index].load_mask(0)
    pmask_shape = pmask.shape
    ##}}

    lfunc = lambda ndx: int(masks[index].class_names[ndx])
    ndx2dfunc = np.vectorize(lfunc)
    ##{{MM:mask os already one-hot-coded}}
    ##gt = np.float32(helpers.one_hot_it(label=gt, label_values=label_values))
    pmask = helpers.reverse_one_hot(pmask)
    ##pmask_dstname = np.zeros(pmask.shape)
    pmask_dstname = np.full(pmask.shape, conf.LST_CLASS + 1)
    pmask_dstname[pmask>0] = ndx2dfunc(pmask[pmask>0])

    return name, pmask_dstname





print('Loading model checkpoint weights')
print('(%s)'%(args.checkpoint_path))
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

maxnparallelworkers = 10

num_iters = int((len(maskfiles) + maxnparallelworkers - 1) / maxnparallelworkers)

timestart = time.time()

print('\nReading promage...\n')
promage_shape, window, input_promage = readPromage()

input_promage_batch = []
print('Predicting on promage...\n')
input_promage_batch.append(np.expand_dims(input_promage, axis=0))
input_promage_batch = input_promage_batch[0]
output_probs_batch = sess.run(probs,feed_dict={net_input:input_promage_batch})
output_probs = output_probs_batch[0][window[0]:window[2],window[1]:window[3],:]

print('Converting prediction to distances...\n')
if MEANDIST:
    lfunc0 = lambda dval: promage.class_names.index(str(dval))
    d2ndxfunc0 = np.vectorize(lfunc0)
    dstname = np.round( np.dot( output_probs[:,:,1:], 
                                np.take(promage.class_names,np.arange(1,num_classes)).astype(int) )
                      ).astype(int)
    clsndx = np.zeros(dstname.shape)
    clsndx[dstname>0] = d2ndxfunc0(dstname[dstname>0])
    clsprb = np.squeeze(
        np.take_along_axis(output_probs, clsndx[:,:,np.newaxis], axis=-1),
        axis=-1);
    dstname[output_probs[:,:,0]>clsprb] = conf.LST_CLASS + 1
    dstname[dstname==0] = conf.LST_CLASS + 1
    dstname[clsprb<args.prb] = 0
else:
    lfunc0 = lambda ndx: int(promage.class_names[ndx])
    ndx2dfunc0 = np.vectorize(lfunc0)
    clsndx = np.argmax(output_probs, axis=-1)
    clsprb = np.squeeze(
        np.take_along_axis(output_probs, clsndx[:,:,np.newaxis], axis=-1),
        axis=-1);
    dstname = np.full(clsndx.shape, conf.LST_CLASS + 1)
    dstname[clsndx>0] = ndx2dfunc0(clsndx[clsndx>0])
    dstname[clsprb<args.prb] = 0


## mask subdiagonals (matching sequence separation) for excluding from consideration
np.fill_diagonal(dstname,0)
for sub in range(1,4):
    np.fill_diagonal(dstname[sub:],0)
    np.fill_diagonal(dstname[:,sub:],0)

## get the region of interest
if args.range == 'upper':
    dstname_vnt = np.triu(dstname)
elif args.range == 'lower':
    dstname_vnt = np.tril(dstname)
else:
    dstname_vnt = np.copy(dstname)


print('Assessing masks wrt prediction...\n')

fp = open(args.out,'w')

fp.write('PFRMAT QA\n')
fp.write('TARGET %s\n'%(args.target))
fp.write('AUTHOR 1929-2214-0552\n') ##ROPIUS0 registration code in CASP14
fp.write('REMARK Error estimate is CB-CB distance in Angstroms\n')
fp.write('METHOD Restraint-Oriented Protocol for Inference and \n')
fp.write('METHOD Understanding of protein Structures.\n')
fp.write('METHOD Based on COMER, Rosetta, and deep learning.\n')
fp.write('MODEL  %d\n'%(args.modelnum))
fp.write('QMODE  2\n')

Xvalue = 0

for i in range(num_iters):
    nparallelworkers = len(maskfiles) - i * maxnparallelworkers
    if maxnparallelworkers < nparallelworkers: nparallelworkers = maxnparallelworkers
    pmaskdata = Parallel(n_jobs=nparallelworkers)(delayed(readMask)(i, j, maxnparallelworkers) for j in range(nparallelworkers))
    pnames, pmasks_dstnames = zip(*pmaskdata)

    for k in range(nparallelworkers):

        name = pnames[k]
        pmask_dstname = pmasks_dstnames[k]

        if dstname.shape != pmask_dstname.shape:
            sys.stderr.write('\nERROR: Inconsistent shape of %s: %s vs %s. Skipped.\n\n'%(
                name,pmask_dstname.shape,dstname.shape))
            continue

        ##output_probs[:,:,:] = 0 ##TEST:
        ##pmask_dstname_tst = pmask_dstname - 1 ##TEST:
        ##pmask_dstname_tst[pmask_dstname_tst==conf.LST_CLASS] = 0 ##TEST:
        ##np.put_along_axis(output_probs, pmask_dstname_tst[:,:,np.newaxis], 1, axis=-1) ##TEST:

        ## exclude rows and columns that have no other values than background (possibly no structure prediction?)
        pmask_rows = np.all(pmask_dstname==(conf.LST_CLASS+1), axis=-1)
        pmask_cols = np.all(pmask_dstname==(conf.LST_CLASS+1), axis=0)
        #pmask_dstname[pmask_rows,:] = Xvalue
        #pmask_dstname[:,pmask_cols] = Xvalue

        #absv = np.abs(pmask_dstname[(dstname_vnt>0) & (dstname_vnt<=args.dst) & (pmask_dstname>Xvalue)] - 
        #                dstname_vnt[(dstname_vnt>0) & (dstname_vnt<=args.dst) & (pmask_dstname>Xvalue)])

        errors_respec_str = ''

        n_res = 0
        mae_mean_global_dst = 0

        for row in range(pmask_dstname.shape[0]):
            if pmask_rows[row] or pmask_cols[row]:
                ## ignore rows and columns completely filled with background values
                errors_respec_str += ' X'
                continue

            pmask_dstname_row_lw = pmask_dstname[row,:row]; pmask_dstname_row_up = pmask_dstname[row,row+1:]
            pmask_dstname_col_lw = pmask_dstname[row+1:,row]; pmask_dstname_col_up = pmask_dstname[:row,row]
            dstname_vnt_row_lw = dstname_vnt[row,:row]; dstname_vnt_row_up = dstname_vnt[row,row+1:]
            dstname_vnt_col_lw = dstname_vnt[row+1:,row]; dstname_vnt_col_up = dstname_vnt[:row,row]

            absv_lower = \
                np.concatenate((
                    np.abs(pmask_dstname_row_lw[(dstname_vnt_row_lw>0) & (dstname_vnt_row_lw<=args.dst)] - 
                             dstname_vnt_row_lw[(dstname_vnt_row_lw>0) & (dstname_vnt_row_lw<=args.dst)]),
                    np.abs(pmask_dstname_col_lw[(dstname_vnt_col_lw>0) & (dstname_vnt_col_lw<=args.dst)] - 
                             dstname_vnt_col_lw[(dstname_vnt_col_lw>0) & (dstname_vnt_col_lw<=args.dst)])
                ))
            absv_upper = \
                np.concatenate((
                    np.abs(pmask_dstname_row_up[(dstname_vnt_row_up>0) & (dstname_vnt_row_up<=args.dst)] - 
                             dstname_vnt_row_up[(dstname_vnt_row_up>0) & (dstname_vnt_row_up<=args.dst)]),
                    np.abs(pmask_dstname_col_up[(dstname_vnt_col_up>0) & (dstname_vnt_col_up<=args.dst)] - 
                             dstname_vnt_col_up[(dstname_vnt_col_up>0) & (dstname_vnt_col_up<=args.dst)])
                ))

            if args.range == 'upper':
                absv = absv_upper
            elif args.range == 'lower':
                absv = absv_lower
            else:
                absv = np.concatenate((absv_lower, absv_upper))

            n_dst = absv.shape[0]

            if n_dst < 1:
                errors_respec_str += ' X'
                continue

            ## divide by expected random distance difference (63*62/2 *1/63) [[0..62]] 
            ## (when not considering distance 65 which is encoded those greater than 64) and 
            ## multiply by 8 so that expected distance error is 8:
            ## (multiply by 16=8*2 when in case of expectation is much lower when using 
            ##  distance and/or probability thresholds)
            mae_mean_dst = np.mean(absv) / (32./(8.*2.))
            ##print(row, absv, ' %.3f %.3f'%(np.mean(absv), mae_mean_dst))

            errors_respec_str += ' %.1f'%(mae_mean_dst)

            mae_mean_global_dst += mae_mean_dst

            n_res += 1

        if n_res:
            ## divide by 8 to constrain quality estimate within the range [0,1]:
            mae_mean_global_dst = 1 - mae_mean_global_dst/(n_res * 8)

        if mae_mean_global_dst < 0 or 1 < mae_mean_global_dst:
            sys.stderr.write('\nWARNING: Invalid mean global distance for %s (%g) adjusted.\n\n'%(
                name,mae_mean_global_dst))
            if mae_mean_global_dst < 0: mae_mean_global_dst = 0
            if 1 < mae_mean_global_dst: mae_mean_global_dst = 1

        fp.write('%s %.5f  %s\n'%(name, mae_mean_global_dst, errors_respec_str))

    ##break ##TEST:

fp.write('END\n')
fp.close()

elapsed = time.time() - timestart

print('\nElapsed: %s\n'%(time.strftime("%H:%M:%S", time.gmtime(elapsed))))

