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
import ray
import multiprocessing
##from joblib import Parallel, delayed
#from scipy.sparse import dok_matrix
from Bio import Align
from Bio.Align import substitution_matrices
from optparse import OptionParser

from mrcnn.config import Config
from mrcnn import utils
#import mrcnn.model as mrclib
#from mrcnn import visualize
#from mrcnn.model import log

## private imports
##sys.path.insert(1, '/data/installed-software/ROPIUS0/bin')
from getchainseq import getChainSequence
from getdist import getDistances

MYMODNAME = os.path.basename(sys.modules[__name__].__file__)
n_cores = multiprocessing.cpu_count()//2 ##use physical cores
ray.init(num_cpus=n_cores)

## ===========================================================================

description = ("Prepare mutually compatible (wrt position) promages and "+
    "structure masks from COMER2 profiles, xcov files, and PDB structures. "+
    "This is a ray version for parallelization, which is much more "+
    "effective than Prallel.")

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)
    parser.add_option("--prodir", dest="prodir", default="", metavar="DIR",
                      help="Directory of COMER2 profiles [required].")
    parser.add_option("--covdir", dest="covdir", default="", metavar="DIR",
                      help="Directory of COMER2 .cov files corresponding COMER2 "+
                           "profiles (names should match). If not given, promages "+
                           "will bot be generated.")
    parser.add_option("--pdbdir", dest="pdbdir", default="", metavar="DIR",
                      help="Directory of PDB structure files (names should match "+
                           "profile names). If not given, structure masks "+
                           "will bot be generated.")
    parser.add_option("--outpmg", dest="outpmg", default="", metavar="DIR",
                      help="Output directory of generated promages [required if "+
                           "promages to be generated].")
    parser.add_option("--outmsk", dest="outmsk", default="", metavar="DIR",
                      help="Output directory of generated struct. masks [required "+
                           "if masks to be generated].")
    parser.add_option("--printaln", action="store_true", dest="printaln",
                      help="Print sequence-structure alignment in the standard output.")
    ret = 1
    (options, args) = parser.parse_args()
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
    """Configuration for testing on/applying to a profile image dataset.
    Derives from the base Config class and overrides values specific to
    profile image dataset.
    """
    # configuration name
    NAME = "promages"
    # configuration application type
    TYPE = "test"


    # Number of classes (including background)
    NUM_CLASSES = 1 + 63  # background + 63 distance bins (2,3,...,64)
    FST_CLASS = 2 # name (distance) of the first/starting class
    LST_CLASS = 64 # name (distance) of the last class


    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = getNXcovfileAttrs() + getNProfileAttrs() * 2 # NOTE: number of promage features/channels
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
            


## ===========================================================================
## Dataset definition
##

@ray.remote
def makeandsavePromageWithoutMask(self, index, npromages, step):
    """Make a promage corresponding to the given name for data at 
    location index in the dataset and write it to file.
    Mask is not made and written.
    """
    for i in range(index, npromages, step):
        name = self.promage_names[i]

        pmgpathname = os.path.join(self.pathpmg, name)
        if os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            continue

        self.preparePromageWithoutMask(i)
        if not os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            np.savez_compressed(pmgpathname+self.pmgext,
                     promage=self.prepared_promages[i])
        # release resources
        self.prepared_promages[i] = None


@ray.remote
def makeandsavePromage(self, index, npromages, step):
    """Make a promage corresponding to the given name for data at 
    location index in the dataset and write it to file.
    """
    for i in range(index, npromages, step):
        name = self.promage_names[i]

        pmgpathname = os.path.join(self.pathpmg, name) if self.pathpmg else name
        mskpathname = os.path.join(self.pathmsk, name) if self.pathmsk else name
        if os.path.isfile(pmgpathname+self.pmgext+self.npzext) and \
           os.path.isfile(mskpathname+self.mskext+self.npzext):
            continue

        if os.path.isfile(mskpathname+self.mskext+self.npzext) and not self.pathpmg:
            continue

        self.preparePromage(i)
        if not self.prepared_promages[i] is None and \
           not os.path.isfile(pmgpathname+self.pmgext+self.npzext):
            np.savez_compressed(pmgpathname+self.pmgext,
                     promage=self.prepared_promages[i])
        if not self.prepared_masks[i] is None and \
           not os.path.isfile(mskpathname+self.mskext+self.npzext):
            np.savez_compressed(mskpathname+self.mskext,
                     promask=self.prepared_masks[i])
                     ##classids=self.prepared_classids[i])
        # release resources
        self.prepared_promages[i] = None
        self.prepared_masks[i] = None
        self.prepared_classids[i] = None


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


    def __init__(self, config, class_map=None, train=False, val=False, keepinmemory=False, printaln=False):
        super().__init__(class_map)
        self.config = config
        self.promage_names = []
        self.pathpro = None
        self.pathcov = None
        self.pathpdb = None
        self.pathpmg = None
        self.pathmsk = None

        self.keepinmemory = keepinmemory
        self.classdict = 'class_dict.csv'

        self.printaln = printaln

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



    def loadPromages(self, prodir, covdir=None, pdbdir=None, outpmg=None, outmsk=None, make=True):
        """Read the list of profiles, profile cross-covariances, and target pdb 
        structures (filenames must match) to define a dataset. If write is True,
        preprocess inputs and outputs and write them to files in binary format for 
        speeding up loading of promages. If write is False, preprocessing will take 
        place on the fly in load_image().
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
        covfiles = []
        if covdir:
            covfiles = self.getFiles(covdir, self.covext)
        if pdbdir:
            pdbfiles = self.getFiles(pdbdir, self.pdbext)

        if np.shape(profiles)[0] < 1:
            sys.exit('ERROR: '+mydefname+'No profiles.')

        if pdbdir and \
           np.shape(profiles)[0] != np.shape(pdbfiles)[0]:
            sys.exit('ERROR: '+mydefname+'Inconsistent data.')

        if covdir and \
           np.shape(profiles)[0] != np.shape(covfiles)[0]:
            sys.exit('ERROR: '+mydefname+'Inconsistent data.')

        if covdir:
            for pf in profiles:
                if not pf in covfiles:
                    sys.stderr.write('ERROR: '+mydefname+'Missing data: '+pf+'\n')
                    sys.exit()
        if pdbdir:
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
        self.pathpmg = outpmg
        self.pathmsk = outmsk

        self.prepared_promages = [None] * npromages
        self.prepared_masks = [None] * npromages
        self.prepared_classids = [None] * npromages

        self.prepare()

        if self.pathmsk is None:
            sys.stderr.write('ERROR: '+mydefname+'No directory for masks given\n')
            sys.exit()
            return

        if covdir and self.pathpmg and not os.path.exists(self.pathpmg):
            os.mkdir(self.pathpmg)

        if not os.path.exists(self.pathmsk):
            os.mkdir(self.pathmsk)

        if not make: return

        # parallel version
        if pdbdir:
            dummy = [makeandsavePromage.remote(self, i, npromages, n_cores) for i in range(n_cores)]
            ##Parallel(n_jobs=n_cores, prefer="threads")(delayed(self.makeandsavePromage)(i, profiles[i]) for i in range(npromages))
        else:
            dummy = [makeandsavePromageWithoutMask.remote(self, i, npromages, n_cores) for i in range(n_cores)]
            ##Parallel(n_jobs=n_cores, prefer="threads")(delayed(self.makeandsavePromageWithoutMask)(i, profiles[i]) for i in range(npromages))
        ray.get(dummy)



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
            pmgfilename = os.path.join(self.pathpmg, name+self.pmgext+self.npzext)
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
        elif self.pathmsk is not None:
            mskfilename = os.path.join(self.pathpmg, name+self.mskext+self.npzext)
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
        covfile_pathname = os.path.join(self.pathcov, name+self.covext) if self.pathcov else ''
        pdbfile_pathname = os.path.join(self.pathpdb, name+self.pdbext)

        if not os.path.isfile(profile_pathname):
            sys.exit('ERROR: '+mydefname+'Profile not found: '+profile_pathname)
        if covfile_pathname and not os.path.isfile(covfile_pathname):
            sys.exit('ERROR: '+mydefname+'Xcov file not found: '+covfile_pathname)
        if not os.path.isfile(pdbfile_pathname):
            sys.exit('ERROR: '+mydefname+'PDB file not found: '+pdbfile_pathname)

        prodct = readProfile(profile_pathname)
        proseq = ''.join(prodct['seq'])
#        print(proseq)
#        with np.printoptions(precision=3, suppress=True):
#            [print(x) for x in prodct['pro']]
        if covfile_pathname:
            procov = readXCovfile(covfile_pathname)
            if len(procov) != len(prodct['pro']):
                sys.exit('ERROR: Inconsistent profile and xcov data.')
#            with np.printoptions(precision=3, suppress=True):
#                for i in range(len(procov)):
#                    for j in range(i,len(procov)):
#                        print(i,j,procov[i,j])

        # get pdb sequence of the first chain EXcluding HETATM records;
        # th result is a dictionary with the pdbstr['seq'] and pdbstr['num']
        # fields
        try:
            pdbstr = getChainSequence(pdbfile_pathname, nohetatm=True)
        #except ValueError as e:
        except:
            sys.stderr.write("ERROR: {}Failed to get structure chain for {}: {}\n".
                format(mydefname, name, sys.exc_info()[0])) ##str(e)))
            self.prepared_promages[image_id] = None
            self.prepared_masks[image_id] = None
            self.prepared_classids[image_id] = None
            return
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

        # substitute all non-canonical residues with X
        pdbstr['seq'] = re.sub('[^ARNDCQEGHILKMFPSTWYVX]', 'X', pdbstr['seq'])
        try:
            alnm = self.alignSeqStr(proseq, pdbstr['seq'])
        except ValueError as e:
            sys.stderr.write(
                "ERROR: {}Failed to align sequence with structure for {}: {}\npdbstr: {}\n\n".
                format(mydefname, name, str(e), pdbstr['seq']))
            self.prepared_promages[image_id] = None
            self.prepared_masks[image_id] = None
            self.prepared_classids[image_id] = class_ids
            return

        alnm1splt = re.split(r'\s+', str(alnm))
        proseq_aln = alnm1splt[0].rstrip()
        pdbseq_aln = alnm1splt[2].rstrip()
        if self.printaln:
            print(proseq_aln, len(proseq_aln))
            print(pdbseq_aln, len(pdbseq_aln))
            print(alnm.aligned)

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
            if proseq_aln[i] in ('-'):
                # do not consider deletion positions in profile;
                # the length of the promage and the mask should match 
                # that of the profile
                continue
            if pdbseq_aln[i] != '-' and not pdbstr['num'][pdbresndx] in pdbres:
                pass #do nothing; test only
            efflen += 1
            ##print(proseq_aln[i],pdbseq_aln[i],pdbstr['num'][pdbresndx])

        if efflen != len(prodct['pro']):
            sys.exit('ERROR: {}Effective length does not match the profile length ({}).'.
                format(mydefname,name))

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
        if covfile_pathname:
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

            if proseq_aln[i] in ('-'):
                continue

            resnumi = pdbstr['num'][pdbresndxi] if pdbseq_aln[i] != '-' else ''

            for j in range(i,len(proseq_aln)):
                if proseq_aln[j] != '-': prondxj += 1
                if pdbseq_aln[j] != '-': pdbresndxj += 1
                if proseq_aln[j] in ('-'):
                    continue

                resnumj = ''
                if pdbseq_aln[i] != '-' and pdbseq_aln[j] != '-': resnumj = pdbstr['num'][pdbresndxj]

                dst = -1
                if pdbseq_aln[i] == '-' or pdbseq_aln[j] == '-':
                    # if the structure lacks a fragment (-), mask will keep background class for it
                    dst = -1
                elif (not resnumi in pdbres) or (not resnumj in pdbres):
                    dst = -1
                elif resnumi == resnumj:
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
                if covfile_pathname:
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
                    if covfile_pathname:
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

        self.prepared_promages[image_id] = promage if covfile_pathname else None
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
            for j in range(i,len(proseq)):
                prondxj += 1
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





## ===========================================================================
##
if __name__ == '__main__':
    ret, options = ParseArguments()
    if not ret: sys.exit()

    config = PromageConfig()

    if not options.prodir or not os.path.exists(options.prodir):
        sys.stderr.write("\nInput directory of profiles not found.\n")
        sys.exit()

    if options.pdbdir:
        if not os.path.exists(options.pdbdir):
            sys.stderr.write("\nInput directory of structures not found.\n")
            sys.exit()
        if not options.outmsk:
            sys.stderr.write("\nOutput directory for masks not specified.\n")
            sys.exit()
        if options.covdir and not os.path.exists(options.covdir):
            sys.stderr.write("\nDirectory of xcov files not found: "+options.covdir+"\n")
            sys.exit()
        if options.covdir and not options.outpmg:
            sys.stderr.write("\nDirectory for output promage files not specified.\n")
            sys.exit()
        promages = PromageDataset(config, printaln=options.printaln)
        promages.loadPromages(prodir=options.prodir, covdir=options.covdir,
            pdbdir=options.pdbdir, outpmg=options.outpmg, outmsk=options.outmsk)
    else:
        if not options.covdir or not os.path.exists(options.covdir):
            sys.stderr.write("\nInput directory of xcov files not found.\n")
            sys.exit()
        if not options.outpmg:
            sys.stderr.write("\nOutput directory for promage files not specified.\n")
            sys.exit()
        promages = PromageDataset(config)
        promages.loadPromages(prodir=options.prodir, covdir=options.covdir, outpmg=options.outpmg)

#<<>>
