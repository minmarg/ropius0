#!/usr/bin/env python
import os,time,cv2, sys, math
import argparse
import numpy as np

sys.path.insert(1, '/data/installed-software/ROPIUS0/bin')
sys.path.insert(1, '/data/installed-software/Semantic-Segmentation-Suite')
from utils import utils, helpers

# private imports
import promage4segm_519 as pmg

parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str, default=None, required=True, help='The promage mask of interest. ')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The dataset being used.')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\nDataset -->", args.dataset)
print("Num Classes -->", num_classes)
print("Image -->", args.mask)

conf = pmg.PromageConfig()

promage_infer = pmg.PromageDataset(conf, keepinmemory=True)
promage_infer.loadPromage1(args.mask)
loaded_mask, _ = promage_infer.load_mask(0)
output_image = loaded_mask
output_image = output_image[:,:,:32] ##<<<
output_valid_indices = np.any(output_image,axis=-1) ##<<<
output_image[~output_valid_indices] = np.concatenate(([1],np.repeat(0,31))) ##<<<

print(output_image.shape)
file_name = utils.filepath_to_name(args.mask)

output_image = helpers.reverse_one_hot(output_image)
out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
cv2.imwrite("%s.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("Finished!")
print("Wrote image " + "%s.png"%(file_name))
