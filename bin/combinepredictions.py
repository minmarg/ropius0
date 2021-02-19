#!/usr/bin/env python
##(C)2019-2021 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University
import sys, os, re
import fnmatch
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from optparse import OptionParser

description = "Make consistent representation of multiple predictions by placing them into one file."
prbext = '.prb';



def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="Comma-separated filenames of model distance predictions.",
                      metavar="FILE(s)")
    parser.add_option("-d", "--indir", dest="indir",
                      help="Comma-separated directories of files with the same "+
                      "filenames and corresponding distance predictions. Filenames "+
                      "will be read form the first directory. "+
                      "(Alternative to option -i)", metavar="DIR(s)")
    parser.add_option("--sep", dest="sep", type=int, default=6,
                      help="Minimum sequence separation; default=%default", metavar="LENGTH")
    parser.add_option("-o", "--output", dest="output",
                      help="output file/directory of files of compiled predictions.", 
                      metavar="FILE/DIR")
    parser.add_option("--cpus", dest="cpus", type=int, default=multiprocessing.cpu_count()//2,
                      help="Number of CPUs to use; default=%default", metavar="CPUs")

    (options, args) = parser.parse_args()

    if options.sep < 1:
        sys.stderr.write("ERROR: Invalid sequence separation given: "+str(options.sep)+"\n")
        sys.exit()

    if options.cpus < 1:
        sys.stderr.write("ERROR: Invalid number of CPUs specified: "+str(options.cpus)+"\n")
        sys.exit()

    return options



def getFilelist(directory, ext):
    """Read `directory' for predictions file with extension `ext', sort them by 
    size and return the resulting list.
    """
    prbfiles = []

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and fnmatch.fnmatch(entry, '*' + ext):
                prbfiles.append(entry.name) ##add filename with extension

    for i in range(len(prbfiles)):
        prbfiles[i] = (prbfiles[i], os.path.getsize(os.path.join(directory,prbfiles[i])))

    #sort files by size
    prbfiles.sort(key=lambda name: name[1], reverse=True)

    for i in range(len(prbfiles)):
        prbfiles[i] = prbfiles[i][0]

    return prbfiles



def getPredictionsFromFile(filename, dsetindex):
    """Read from the given file the distance predictions made by a model;
    dsetindex, data set index (index of an input directory of files).
    """
    mydefname = 'getPredictionsFromFile: '
    if not os.path.isfile(filename):
        sys.exit('ERROR: '+mydefname+'Predictions file not found: '+ filename)

    dstlist = []
    erc = 1
    pos = 0

    with open(filename) as fp:
        for line in fp:
            flds = re.split(r'\s+', line.strip())
            pos += 1
            if len(flds) < 4:
                sys.stderr.write('ERROR: Insufficient number of fields at distance file position %d\n'%(pos))
                erc = 0
                break
            dstlist.append(flds[0:4]+[dsetindex])

    return [erc, dstlist]



def combinePredictions(dstpreds, inputname, outfilename):
    """Combine multiple predictions consistently and write the 
    resulting combination to file.
    """
    mydefname = 'getPredictionsFromFile: '
    n_dstpreds = len(dstpreds)

    if not n_dstpreds:
        sys.exit('ERROR: '+mydefname+'No distance predictions.')

    npdstpreds = np.concatenate((dstpreds[:])).astype(float);
    ##npdstpreds = npdstpreds[(npdstpreds[:,0]*100000+npdstpreds[:,1]).argsort()]##NOTE: errors may emerge for large indices
    npdstpreds = npdstpreds[np.lexsort((npdstpreds[:,1],npdstpreds[:,0]))]##sorts by the order of fields (0,1!) in which they appear in the structure

    sys.stderr.write('%s: %d prediction(s) (shape: %s)\n'%(inputname,n_dstpreds,str(npdstpreds.shape)))

    bckpred = np.zeros(len(npdstpreds[0,4:])).astype(float) ##background prediction values
    bckpred[-1] = 1

    ##print(bckpred)

    with open(outfilename,'w') as ofile:

        lid = 0
        cmbstr = '#R1 R2   ';
        cmbstr += '    Dst  Prob' * n_dstpreds
        ofile.write('%s\n'%(cmbstr));
        while True:
            if len(npdstpreds) <= lid:
                break
            rid1 = int(npdstpreds[lid,0])
            rid2 = int(npdstpreds[lid,1])
            if rid1 < 1 or rid2 < 1 or rid2 <= rid1:
                sys.exit('ERROR: Invalid residue number at distance file position %d (%s: %d %d).'%(
                    lid,inputname,rid1,rid2))

            dfld = [[-1,-1]] * n_dstpreds
            dfld[npdstpreds[lid,4].astype(int)] = npdstpreds[lid,2:4]
            while True:
                lid += 1
                if len(npdstpreds) <= lid or \
                   int(npdstpreds[lid,0]) != rid1 or int(npdstpreds[lid,1]) != rid2:
                    break
                dfld[npdstpreds[lid,4].astype(int)] = npdstpreds[lid,2:4]

            if abs(rid2 - rid1) < options.sep:
                continue

            ridstr = '%d %d'%(rid1,rid2)
            cmbstr = ''
            for i in range(len(dfld)):
                cmbstr += '   %4s %5s'%('----','-----') if dfld[i][0] < 0 else '   %4.1f %5.3f'%(dfld[i][0],dfld[i][1])

            ##write distance probabilities
            ofile.write('%-9s%s\n'%(ridstr,cmbstr))



def processPredictionFiles(infiles, outputfile):
     """Process prediction files supposed to describe the same object.
     The file list is given with infiles. Write the result to file. 
     """
     dstpreds = []
     for n,fl in enumerate(infiles):
         code, dsts = getPredictionsFromFile(fl, n)
         if not code:
             sys.exit('ERROR: Reading model distance predictions failed: %s'%(fl))
         #NOTE: sizes might differ
         dstpreds.append(dsts)

     bname = os.path.basename(infiles[0])
     inputname = os.path.splitext(bname)[0]
     combinePredictions(dstpreds, inputname, outputfile)



if __name__ == "__main__":
    options = ParseArguments()

    if options.input:

        if not options.output:
            sys.exit('ERROR: Output filename is not provided.')
        elif os.path.isdir(options.output):
            sys.exit("ERROR: Output file is a directory: %s"%(options.output))

        infiles = re.split(r',', options.input.strip())
        processPredictionFiles(infiles, options.output)

    elif options.indir:

        if not os.path.exists(options.output):
            os.mkdir(options.output)
        elif os.path.isfile(options.output):
            sys.exit("ERROR: Output directory is a file: %s"%(options.output))

        indirs = re.split(r',', options.indir.strip())

        if not len(indirs): sys.ext()

        if not (os.path.exists(indirs[0]) and os.path.isdir(indirs[0])):
            sys.exit("ERROR: Input directory not found: %s"%(indirs[0]))

        prbfiles1 = getFilelist(indirs[0], prbext)

        collection = []
        outfiles = []

        for _,f in enumerate(prbfiles1):
            flst = [os.path.join(indirs[0], f)]
            for d in range(1,len(indirs)):
                cndfile = os.path.join(indirs[d], f)
                if os.path.isfile(cndfile): flst.append(cndfile)
            collection.append(flst)
            outfiles.append(os.path.join(options.output, f))

        # parallel execution
        n_cores = options.cpus
        if n_cores < 1: n_cores = 1

        Parallel(n_jobs=n_cores)(delayed(processPredictionFiles)(
            collection[i], outfiles[i]) for i in range(len(collection)))

#<<>>
