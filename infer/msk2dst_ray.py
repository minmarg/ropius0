#!/usr/bin/env python

## (C)2021 Mindaugas Margelevicius, Vilnius University

import sys, os
import fnmatch
import argparse
import numpy as np
import multiprocessing
import ray

# private imports
import promage4segm_519 as pmg

parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str, default=None, required=False, help='The promage mask of interest. ')
parser.add_argument('--indir', type=str, default=None, required=False, help='Input directory of promage masks. ')
##parser.add_argument('--dataset', type=str, default=None, required=True, help='The dataset being used.')
parser.add_argument('--output', type=str, default=None, required=True, help='Output file/directory of distances between residues. ')
args = parser.parse_args()

if not (args.mask or args.indir):
    sys.exit("ERROR: Mask file or directory of masks should be provided.")

n_cores = multiprocessing.cpu_count()//2 ##use physical cores

ray.init(num_cpus=n_cores)

##class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

##num_classes = len(label_values)

##sys.stderr.write("Dataset: " + args.dataset + "\n")
##sys.stderr.write("Number of classes: %d\n"%(num_classes))
if args.mask: sys.stderr.write("Mask: " + args.mask + "\n")

conf = pmg.PromageConfig()

mskext = '.msk.npz' ## file extension of compressed preprocessed promage masks with its class ids in binary format

@ray.remote
def writeDistances(index, nmskfiles, step, mskfiles_id, indir, outdir):
    """Convert information in the mask file to distances;
    inputfile, input file; outputfile, output file.
    """
    def reverse_one_hot(image):
        """
        Transform a 2D array in one-hot format (depth is num_classes),
        to a 2D array with only 1 channel, where each pixel value is
        the classified class key. (code adopted from Semantic-Segmentation-Suite.)
        """
        x = np.argmax(image, axis = -1)
        return x

    for i in range(index, nmskfiles, step):
        inputfile = os.path.join(indir, mskfiles_id[i])
        outputfile = os.path.join(outdir, mskfiles_id[i]+'_distances.out')

        if os.path.isfile(outputfile):
            continue

        promage_infer = pmg.PromageDataset(conf, keepinmemory=True)
        promage_infer.loadPromage1(inputfile)
        loaded_mask, _ = promage_infer.load_mask(0)

        sys.stderr.write("Mask shape: " + str(loaded_mask.shape) + "\n")

        ## translate one-hot-coded mask to 2D matrix of distance values
        lfunc = lambda ndx: int(promage_infer.class_names[ndx])
        ndx2dfunc = np.vectorize(lfunc)
        ## the mask is one-hot-coded, reverse it to distance values
        loaded_mask = reverse_one_hot(loaded_mask)
        ## fill the matrix with a value one greater than the maximum distance
        dstmtx = np.full(loaded_mask.shape, conf.LST_CLASS + 1)
        if np.shape(loaded_mask[loaded_mask>0])[0] > 0:
            dstmtx[loaded_mask>0] = ndx2dfunc(loaded_mask[loaded_mask>0])


        ## print the upper triangle of the distance matrix
        with open(outputfile,'w') as ofile:
            for row in range(dstmtx.shape[0]):
                for col in range(row+1,dstmtx.shape[0]):
                    dst = dstmtx[row,col]
                    if dst > conf.LST_CLASS:
                        continue
                    ofile.write("%d %d %5.1f\n"%(row+1, col+1, dst))



if __name__ == '__main__':

    if args.mask:
        infile = args.mask.rsplit(mskext)[0]
        writeDistances(infile, args.output)

    elif args.indir:
        if not os.path.exists(args.indir):
            sys.exit("ERROR: Input directory does not exist: %s"%(args.indir))
        if os.path.isfile(args.indir):
            sys.exit("ERROR: Input directory provided is a file: %s"%(args.indir))

        if not os.path.exists(args.output):
            os.mkdir(args.output)
        elif os.path.isfile(args.output):
            sys.exit("ERROR: Output directory is a file: %s"%(args.output))

        mskfiles = []

        with os.scandir(args.indir) as entries:
            for entry in entries:
                #if entry.is_file() and (fnmatch.fnmatch(entry, '*2GNX_A' + ext) or fnmatch.fnmatch(entry, '*4BEG_A' + ext)):
                #if entry.is_file() and (fnmatch.fnmatch(entry, '*5J*' + ext) or fnmatch.fnmatch(entry, '*4BEG_A' + ext)):
                if entry.is_file() and fnmatch.fnmatch(entry, '*' + mskext):
                    mskfiles.append(entry.name.rsplit(mskext)[0]) ##add filename only

        # parallel version
        mskfiles_id = ray.put(mskfiles) 
        dummy = [writeDistances.remote(i, len(mskfiles), n_cores, mskfiles_id, args.indir, args.output) for i in range(n_cores)]
        ray.get(dummy)

    sys.stderr.write("Finished!\n")

