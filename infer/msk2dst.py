#!/usr/bin/env python

## (C)2020 Mindaugas Margelevicius, Vilnius University

import sys, os
import argparse
import numpy as np

sys.path.insert(1, '/data/installed-software/ROPIUS0/bin')
sys.path.insert(1, '/data/installed-software/ROPIUS0/infer')
sys.path.insert(1, '/data/installed-software/Semantic-Segmentation-Suite')
from utils import utils, helpers

# private imports
import promage4segm_519 as pmg

parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str, default=None, required=True, help='The promage mask of interest. ')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The dataset being used.')
parser.add_argument('--output', type=str, default=None, required=True, help='Output file of distances between residues. ')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

sys.stderr.write("Dataset: " + args.dataset + "\n")
sys.stderr.write("Number of classes: %d\n"%(num_classes))
sys.stderr.write("Mask: " + args.mask + "\n")

conf = pmg.PromageConfig()


promage_infer = pmg.PromageDataset(conf, keepinmemory=True)
promage_infer.loadPromage1(args.mask)
loaded_mask, _ = promage_infer.load_mask(0)

sys.stderr.write("Mask shape: " + str(loaded_mask.shape) + "\n")


## translate one-hot-coded mask to 2D matrix of distance values
lfunc = lambda ndx: int(promage_infer.class_names[ndx])
ndx2dfunc = np.vectorize(lfunc)
## the mask is one-hot-coded, reverse it to distance values
loaded_mask = helpers.reverse_one_hot(loaded_mask)
## fill the matrix with a value one greater than the maximum distance
dstmtx = np.full(loaded_mask.shape, conf.LST_CLASS + 1)
dstmtx[loaded_mask>0] = ndx2dfunc(loaded_mask[loaded_mask>0])


## print the upper triangle of the distance matrix
with open(args.output,'w') as ofile:
    for row in range(dstmtx.shape[0]):
        for col in range(row+1,dstmtx.shape[0]):
            dst = dstmtx[row,col]
            if dst > conf.LST_CLASS:
                continue
            ofile.write("%d %d %5.1f\n"%(row+1, col+1, dst))

sys.stderr.write("Finished!\n")

