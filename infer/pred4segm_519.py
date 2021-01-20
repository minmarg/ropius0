#!/usr/bin/env python
import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

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
parser.add_argument('--promage', type=str, default=None, required=True, help='The promage you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=128, help='Height of cropped input image to network.')
parser.add_argument('--crop_width', type=int, default=128, help='Width of cropped input image to network.')
parser.add_argument('--model', type=str, default='FC-DenseNet103', required=False, help='The model being used.')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend being used.')
parser.add_argument('--dataset', type=str, default="SEMSEGM", required=False, help='The dataset in use.')
parser.add_argument('--outpat', type=str, required=False, help='Output filename pattern.')
parser.add_argument('--mean', action='store_true', help='Calculate the mean predicted distance for each pair of residues.')
parser.add_argument('--printall', action='store_true', help='Save all predicted probabilities instead of binned.')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.promage)
print("Mean -->", args.mean)

# Initializing network
##{{MM
config = tf.ConfigProto(device_count={'GPU':0},gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=str(0)),log_device_placement=False)
##}}
##config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

##{{MM
conf = pmg.PromageConfig()
BATCHSIZE = 32 ##max number of promages to perform simultaneous prediction on
##step in x and y directions for predicting the whole promage;
##NOTE: values <= 0 indicate using no manual incremental prediction
PREDSTEP = 0 #32
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



print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


print("Testing image " + args.promage)

##{{MM
promage_infer = pmg.PromageDataset(conf, keepinmemory=True)
promage_infer.loadPromage1(args.promage)
loaded_image = promage_infer.load_image(0)
if PREDSTEP > 0:
    loaded_image, window, _ = pad_image(loaded_image, min_dim=args.crop_height)
else:
    loaded_image, window, _ = pad64_image(loaded_image)
    args.crop_height = loaded_image.shape[0]
    args.crop_width = loaded_image.shape[1]

##input_image = loaded_image
##input_image = input_image[:args.crop_height,:args.crop_width,:]
##if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
##if NORMALIZE: input_image = input_image / conf.PIXEL_NORM
##input_image = np.expand_dims(input_image, axis=0)
##}}
##loaded_image = utils.load_image(args.image)
##resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
##input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

ndxi_beg = 0
ndxi_end = args.crop_height
ndxj_beg = 0
ndxj_end = args.crop_width
input_image_batch = []
input_image_batch_crops = []
## overlap counts for output image
output_image_ovlps = np.zeros(loaded_image.shape[:2])
## output prediction itself
output_mean = np.zeros(loaded_image.shape[:2])
output_probs = np.zeros(loaded_image.shape[:2]+(num_classes,))

while ndxi_end <= loaded_image.shape[0]:
    ndxj_beg = 0
    ndxj_end = args.crop_height
    while ndxj_end <= loaded_image.shape[1]:
        input_image = loaded_image[ndxi_beg:ndxi_end, ndxj_beg:ndxj_end, :]
        if SUBT_MEAN: input_image = input_image.astype(np.float32) - conf.MEAN_PIXEL
        if NORMALIZE: input_image = input_image / conf.PIXEL_NORM
        input_image_batch.append(np.expand_dims(input_image, axis=0))
        input_image_batch_crops.append([ndxi_beg, ndxi_end, ndxj_beg, ndxj_end])
        if BATCHSIZE <= len(input_image_batch) or \
           (ndxi_end >= loaded_image.shape[0] and ndxj_end >= loaded_image.shape[1]):

            if len(input_image_batch) == 1:
                input_image_batch = input_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))

            #_, output_probs_batch = sess.run([network,probs],feed_dict={net_input:input_image_batch})
            output_probs_batch = sess.run(probs,feed_dict={net_input:input_image_batch})

            print("Batch of %d crops processed"%(len(output_probs_batch)))
            for i in range(len(output_probs_batch)):
##                if MEANDIST:
##                    patch_mean = np.dot( output_probs_batch[i][:,:,1:], np.arange(1,num_classes) )
##                    patch_mean_prbs = np.squeeze(
##                        np.take_along_axis(output_probs_batch[i], 
##                            np.around(patch_mean[:,:,np.newaxis]).astype(int), 
##                            axis=-1), 
##                        axis=-1);
##                    patch_mean[patch_mean_prbs<output_probs_batch[i][:,:,0]] = 0
##                    output_mean[
##                        input_image_batch_crops[i][0]:input_image_batch_crops[i][1],
##                        input_image_batch_crops[i][2]:input_image_batch_crops[i][3]] += \
##                      patch_mean[:,:]
##                    ##np.dot( output_probs_batch[i][:,:,1:], np.arange(1,num_classes) )
##                    ####np.dot( output_probs_batch[i][:,:,1:],
##                    ####        np.array(promage_infer.class_names[1:]).astype(int) )
##                else:
##                    output_mean[
##                      input_image_batch_crops[i][0]:input_image_batch_crops[i][1],
##                      input_image_batch_crops[i][2]:input_image_batch_crops[i][3]] += \
##                    np.argmax( output_probs_batch[i][:,:,:], axis=-1 )
                output_probs[
                    input_image_batch_crops[i][0]:input_image_batch_crops[i][1],
                    input_image_batch_crops[i][2]:input_image_batch_crops[i][3],:] += \
                  output_probs_batch[i][:,:,:]
                output_image_ovlps[
                  input_image_batch_crops[i][0]:input_image_batch_crops[i][1],
                  input_image_batch_crops[i][2]:input_image_batch_crops[i][3]] += 1

            input_image_batch = []
            input_image_batch_crops = []

        if ndxj_end < loaded_image.shape[1]:
            ndxj_beg += PREDSTEP
            ndxj_end = ndxj_beg + args.crop_width
            if ndxj_end > loaded_image.shape[1]:
                ndxj_end = loaded_image.shape[1]
                ndxj_beg = ndxj_end - args.crop_width
        else: break

    if ndxi_end < loaded_image.shape[0]:
        ndxi_beg += PREDSTEP
        ndxi_end = ndxi_beg + args.crop_height
        if ndxi_end > loaded_image.shape[0]:
            ndxi_end = loaded_image.shape[0]
            ndxi_beg = ndxi_end - args.crop_height
    else: break


##output_mean /= output_image_ovlps
output_probs /= output_image_ovlps[:,:,np.newaxis]
output_probs = output_probs[window[0]:window[2],window[1]:window[3]]
output_mean = output_mean[window[0]:window[2],window[1]:window[3]]
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
print(output_mean)

if EEF_NUM_CLASSES < conf.NUM_CLASSES:
    output_mean[output_mean>EEF_NUM_CLASSES-1] = 0 ##<<<


file_name = utils.filepath_to_name(args.promage)
if args.outpat:
    file_name = args.outpat
predfilename = '%s_predmean.png'%(file_name) if MEANDIST else '%s_pred.png'%(file_name)
probfilename = '%s_predmean.prb'%(file_name) if MEANDIST else '%s_pred.prb'%(file_name)

avg2_probs = np.zeros(num_classes)

with open(probfilename,'w') as fp:
    for i in range(output_mean.shape[0]):
        for j in range(i+1,output_mean.shape[0]):
            avg2_probs = (output_probs[i,j,:]+output_probs[j,i,:])*0.5
            if MEANDIST:
                dstname = int(round( np.dot( avg2_probs[1:], np.take(promage_infer.class_names,np.arange(1,num_classes)).astype(int) ) ))
                clsndx = promage_infer.class_names.index(str(dstname)) if 0 < dstname else 0
            else:
                clsndx = np.argmax(avg2_probs)
                dstname = int(promage_infer.class_names[clsndx]) if 0 < clsndx else 0
            clsprb = avg2_probs[clsndx]
            if clsndx == 0 or clsprb < avg2_probs[0]:
                ##either background or background probability is greater than the mean
                continue

            ##clsndx = int(round((output_mean[i,j] + output_mean[j,i]) * 0.5))
            ##if clsndx == 0:
            ##    continue
            ##clsprb = (output_mean_probs[i,j] + output_mean_probs[j,i]) * 0.5
            ##dstname = int(promage_infer.class_names[clsndx])

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


print("")
print("Finished! " + "Output: %s and %s"%(predfilename,probfilename))

