#!/usr/bin/env python

import sys, os, re
import bisect
import numpy as np
from optparse import OptionParser

description = "Convert model distance predictions to CASP format of Residue-Residue distance prediction."

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="Comma-separated filenames of model distance predictions",
                      metavar="FILE(s)")
    parser.add_option("-t", "--target", dest="target",
                      help="Target name", metavar="TARGET")
    parser.add_option("--sep", dest="sep", type=int, default=6,
                      help="Minimum sequence separation; default=%default", metavar="RANGE")
    parser.add_option("--chains", dest="chains", default="",
                      help="Comma-separated chain IDs of the protein predicted", 
                      metavar="CHAINS")
    parser.add_option("--ends", dest="ends", default="",
                      help="Comma-separated end positions of the given chains",
                      metavar="POSITIONS")
    parser.add_option("--skip", action="store_true", dest="skip",
                      help="Skip predictions for the same chain if chains are provided")
    parser.add_option("-o", "--outfile", dest="output",
                      help="output file of distance constraints", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.input and options.target):
        sys.stderr.write("ERROR: Input file and/or target name is not provided.\n")
        sys.exit()

    if options.sep < 0.0:
        sys.stderr.write("ERROR: Invalid sequence separation given: "+str(options.sep)+"\n")
        sys.exit()

    return options



def getPredictions(filename):
    """Read from the given file the distance predictions made by a model.
    """
    mydefname = 'getPredictions: '
    if not os.path.isfile(filename):
        sys.exit('ERROR: '+mydefname+'Predictions file not found: '+ filename)

    dstlist = []
    erc = 1
    pos = 0

    with open(filename) as fp:
        for line in fp:
            flds = re.split(r'\s+', line.strip())
            pos += 1
            if len(flds) != 4 and len(flds) != 14:
                sys.stderr.write('ERROR: Unexpected number of fields at distance file position %d\n'%(pos))
                erc = 0
                break
            dstlist.append(flds)

    return [erc, dstlist]



if __name__ == "__main__":
    options = ParseArguments()

    dstpreds = []
    infiles = re.split(r',', options.input.strip())
    for _,fl in enumerate(infiles):
        code, dsts = getPredictions(fl)
        if not code:
            sys.exit('ERROR: Reading model distance predictions failed: %s'%(fl))
        #NOTE: sizes might differ
        dstpreds.append(dsts)

    n_dstpreds = len(dstpreds)

    if not n_dstpreds:
        sys.exit('ERROR: No distance predictions read.')

    npdstpreds = np.concatenate((dstpreds[:])).astype(float);
    ##npdstpreds = npdstpreds[(npdstpreds[:,0]*100000+npdstpreds[:,1]).argsort()]##NOTE: errors may emerge for large indices
    npdstpreds = npdstpreds[np.lexsort((npdstpreds[:,1],npdstpreds[:,0]))]##sorts by the order of fields (0,1!) in which they appear in the structure

    sys.stderr.write('%d prediction(s) read (shape: %s)\n'%(n_dstpreds,str(npdstpreds.shape)))

    bckpred = np.zeros(len(npdstpreds[0,4:])).astype(float) ##background prediction values
    bckpred[-1] = 1

    print(bckpred)

    chains = []
    chbpos = []
    chepos = []

    if options.chains:
        chains = re.split(r',', options.chains.strip())
        if not options.ends:
            sys.exit('ERROR: Chain end positions not provided.')
        chepos = re.split(r',', options.ends.strip())
        chepos = np.asarray(np.array(chepos).astype(int))
        if len(chains) != len(chepos) or len(chains) < 1:
            sys.exit('ERROR: Inconsistent or invalid chain IDs and their end positions.')
        for c in range(len(chepos)):
            if c and chepos[c] <= chepos[c-1]:
                sys.exit('ERROR: Unsorted chain end positions.')
            chbpos.append(chepos[c-1]+1 if c else 1)

    with open(options.output,'w') as ofile:
        ofile.write('PFRMAT RR\n')
        ofile.write('TARGET %s\n'%(options.target))
        ofile.write('AUTHOR 1929-2214-0552\n') ##ROPIUS0 registration code in CASP14
        ofile.write('METHOD Restraint-Oriented Protocol for Inference and \n')
        ofile.write('METHOD Understanding of protein Structures.\n')
        ofile.write('METHOD Based on COMER, Rosetta, and deep learning.\n')
        ofile.write('RMODE  2\n')
        ofile.write('MODEL  1\n')

        lid = 0
        while True:
            if len(npdstpreds) <= lid:
                break
            rid1 = int(npdstpreds[lid,0])
            rid2 = int(npdstpreds[lid,1])
            if rid1 < 1 or rid2 < 1 or rid2 <= rid1:
                sys.exit('ERROR: Invalid residue number at distance file position %d (%d %d).'%(lid,rid1,rid2))

            dfld = npdstpreds[lid,4:]
            n_dfld = 1
            while True:
                lid += 1
                if len(npdstpreds) <= lid or \
                   int(npdstpreds[lid,0]) != rid1 or int(npdstpreds[lid,1]) != rid2:
                    break
                if len(npdstpreds[lid,4:]) != len(dfld):
                    sys.exit('ERROR: Inconsistent number of fields of prediction '+
                         'file %d at distance file position %d (%d %d).'%(dpit,lid,rid1,rid2))
                dfld += npdstpreds[lid,4:]
                n_dfld += 1
            while n_dfld < n_dstpreds:
                dfld += bckpred #this is dummy prediction for background (no contact, or distance>20)
                n_dfld += 1
            dfld /= n_dfld

            cid1 = cid2 = ''
            beg1 = beg2 = 1

            if len(chains):
                ndxchn1 = bisect.bisect_left(chepos, rid1)
                ndxchn2 = bisect.bisect_left(chepos, rid2)
                if len(chepos) <= ndxchn1 or len(chepos) <= ndxchn2:
                    sys.exit('ERROR: Chain(s) not found for positions at distance file line %d (%d %d).'%(lid-1,rid1,rid2))
                cid1 = chains[ndxchn1]
                beg1 = chbpos[ndxchn1]
                cid2 = chains[ndxchn2]
                beg2 = chbpos[ndxchn2]
                if options.skip and cid1 == cid2:
                    continue

            ##reindex:
            rid1 = rid1 - beg1 + 1
            rid2 = rid2 - beg2 + 1

            if cid1 == cid2 and abs(rid2 - rid1) < options.sep:
                continue

            ##probability for a pair to be in contact (<8A)
            ##NOTE: CASP definition is not accurate: the sum of the first three 
            ## predictions (<=8A) is not necessarily equal to the contact probability
            prbcntct = np.sum(dfld[:3])
            ##write distance probabilities
            ofile.write('%s%d %s%d  %.3g '%(cid1,rid1,cid2,rid2,prbcntct))
            [ofile.write(' %.3g'%(dfld[pi])) for pi in range(len(dfld))]
            ofile.write('\n')

        ofile.write('END\n')

#<<>>
