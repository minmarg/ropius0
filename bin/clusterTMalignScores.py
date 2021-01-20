#!/usr/bin/env python

import sys, os, re
import numpy as np
from sklearn.cluster import DBSCAN
from optparse import OptionParser
from collections import defaultdict
import fnmatch

description = "Cluster TMalign scores."


def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-d", "--inputdir", dest="inputdir",
                      help="Input directory of TMalign summary files", metavar="DIR")
    parser.add_option("-s", "--minscore", dest="minscore", type=float, default=0.7,
                      help="Minimum TMalign score for a pair to put them in the same cluster; "
                      "default=%default", metavar="SCORE")
    parser.add_option("-o", "--outfile", dest="output",
                      help="Output file of cluster memberships", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.inputdir):
        sys.stderr.write("ERROR: Input directory is not provided.\n")
        sys.exit()

    if options.minscore <= 0.0 or 1.0 <= options.minscore:
        sys.stderr.write("ERROR: Invalid TM align score threshold.\n")
        sys.exit()

    if options.inputdir and not os.path.isdir(options.inputdir):
        sys.stderr.write("ERROR: Input directory not found: "+options.inputdir+"\n")
        sys.exit()

    return options


def getFiles(directory, ext=''):
    """Read directory contents; ext, file extension
    """
    files = []
    if not os.path.isdir(directory):
        sys.exit('ERROR: Directory not found: '+ directory)

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and fnmatch.fnmatch(entry, '*' + ext):
                ##files.append(os.path.splitext(entry.name)[0]) ##add filename only
                files.append(entry.name)
    return files


def readSummaryFile(summfile, modelsdict):
    """Read a summary file of TMalign scores;
    modelsdict, dictionary of the scores between the models.
    """
    mydefname = 'readSummaryFile: '
    if not os.path.isfile(summfile):
        sys.exit('ERROR: '+mydefname+'TMalign summary file not found: '+summfile)

    proname = os.path.basename(summfile)
    modelname = ''
    recnum = 0

    with open(summfile) as fp:
        for line in fp:
            recnum += 1
            flds = re.split(r'\s+', line.rstrip())
            if len(flds) != 11 or flds[0][0] == '#': continue
            if not modelname: modelname = flds[0]
            if modelname and modelname != flds[0]:
                sys.exit('ERROR: Invalid first model\'s name in %s (line %d):\n%s\n'%(summfile,recnum,line))
            #modelsdict[flds[0]][flds[1]] = [flds[2]]
            modelsdict[flds[0]].update({flds[1]: flds[2]})


def makeOutput(mnames, tmscoremtx, clustering, minscore):
    """Produce content for output given the original dictionary of models,
    the corresponding TM align score matrix, and the clustering result.
    """
    content = 'Cluster membership score threshold= %s\n'%(minscore)
    labels = clustering.labels_
    cltndxs = np.unique(labels)
    cndxs = np.concatenate((cltndxs[cltndxs>=0],cltndxs[cltndxs<0]))
    for cn in cndxs:
        ssum = lsum = cmax = cx = 0
        nels = 0
        mndxs = np.where(labels==cn)[0]
        for mn in mndxs:
            lsum = np.sum(tmscoremtx[mn,mndxs])
            if cmax < lsum: cmax = lsum; cx = mn
            ssum += lsum
            nels += len(mndxs)
        ssum -= np.sum(tmscoremtx[mndxs,mndxs])
        nels -= len(mndxs)
        if nels < 1: nels = 1
        if cn < 0:
            content += '\n\n\nSINGLETONS:\n\n'
        else:
            content += '\n\nCLUSTER %d (avg TM-score= %.3f; Centre %s):\n\n'%(
                cn, ssum/nels, mnames[cx])
        for i in range(len(mndxs)): content += '%s\n'%(mnames[mndxs[i]])
    return content



if __name__ == "__main__":
    options = ParseArguments()
    files = getFiles(options.inputdir)
    mdict = defaultdict(dict) #{}
    for i in range(len(files)):
        fullfilename = os.path.join(options.inputdir, files[i])
        readSummaryFile(fullfilename, mdict)

    mnames = sorted(mdict)
    for model1 in mnames:
        m1mnames = sorted(mdict[model1])
        if len(mnames) != len(m1mnames):
            sys.exit('ERROR: Inconsistent # model names (missing?) for model %s\n'%(model1))
        for i, model2 in enumerate(m1mnames):
            if model2 != mnames[i]:
                sys.exit('ERROR: Model names do not match over all models; models %s and %s\n'%(model1, model2))

    tmscoremtx = np.array(
        [[mdict[model1][model2] for model2 in sorted(mdict[model1])] for model1 in sorted(mdict)]).astype(float)

    print(tmscoremtx)

    clustering = DBSCAN(eps=1.-options.minscore, min_samples=2, metric='precomputed').fit(1-tmscoremtx)

    ##print(clustering.core_sample_indices_)
    ##print(clustering.components_)
    ##print(clustering.labels_)

    content = makeOutput(mnames, tmscoremtx, clustering, options.minscore)

    if options.output:
        with open(options.output,'w') as ofile:
            ofile.write(content)
    else:
        print(content)

#<<>>
