#!/usr/bin/env python

import sys, os
import re
import numpy as np
from optparse import OptionParser

MAXDST = 64.0

description = "Convert model distance predictions to format of Rosetta constraints."

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="Comma-separated filenames of model distance predictions",
                      metavar="FILE(s)")
    parser.add_option("-s", "--sqfile", dest="sqfile",
                      help="Protein sequence file in FASTA format", metavar="FILE")
    parser.add_option("--beg", dest="beg", type=int, default=1,
                      help="Start position of the sequence fragment to process; " \
                        "default=%default", metavar="POSITION")
    parser.add_option("--end", dest="end", type=int, default=-1,
                      help="End position of the sequence fragment to process; " \
                        "default=[The end of the sequence]", metavar="POSITION")
    parser.add_option("--sep", dest="sep", type=int, default=0,
                      help="Minimum sequence separation; default=%default", metavar="RANGE")
    parser.add_option("--prb", dest="prb", type=float, default=0,
                      help="Minimum probability threshold for converting distance " \
                        "predictions to constraints; default=%default", metavar="PROBABILITY")
    parser.add_option("--dst", dest="dst", type=float, default=-1,
                      help="Ignore distance predictions larger than this value; " \
                        "default=[Do not ignore any values]", metavar="DISTANCE")
    parser.add_option("--profile", dest="profile", default="",
                      help="Ignore distance predictions within secondary structure " \
                        "assignments present in the given profile; Profile length " \
                        "should be consistent with the sequence length", metavar="FILE")
    parser.add_option("-t", "--atom", dest="atom", default="CB",
                      help="Type of atoms between which the distance is to be assigned; " \
                        "default=%default", metavar="ATOM_TYPE")
    parser.add_option("-g", "--glyatom", dest="glyatom", default="CA",
                      help="Type of GLY atom for distance assignment; " \
                        "default=%default", metavar="ATOM_TYPE")
    parser.add_option("-c", "--cst", dest="cstype", default="HARMONIC",
                      help="Type of Rosetta constraints (e.g., -c \"SPLINE EPR_DISTANCE\"); " \
                        "default=%default", metavar="TYPE")
    parser.add_option("-o", "--outfile", dest="output",
                      help="output file of distance constraints", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.input and options.sqfile and options.output):
        sys.stderr.write("ERROR: Input and/or output file is not provided.\n")
        sys.exit()

    if options.sqfile and not os.path.isfile(options.sqfile):
        sys.stderr.write("ERROR: Sequence file not found: "+options.sqfile+"\n")
        sys.exit()

    if options.prb < 0.0 or 1.0 < options.prb:
        sys.stderr.write("ERROR: Invalid probability threshold given: "+str(options.prb)+"\n")
        sys.exit()

    if options.sep < 0.0:
        sys.stderr.write("ERROR: Invalid sequence separation given: "+str(options.sep)+"\n")
        sys.exit()

    return options


def getSequence(filename):
    """Read the sequence present in the given file in FASTA format.
    """
    sqn = ''
    with open(filename) as fp:
        for line in fp:
            txt = line.strip()
            if txt[0] == '>':
                continue
            sqn += txt
    return [1, sqn]


def readProfileSS(filename):
    """Read the secondary structure annotation in the given 
    COMER profile.
    """
    mydefname = 'readProfileSS: '
    if not os.path.isfile(filename):
        sys.exit('ERROR: '+mydefname+'Profile not found: '+ filename)

    ssslist = []
    erc = 1
    cnt = 0
    sscode = -1
    sss = 'C'

    with open(filename) as fp:
        for line in fp:
            flds = re.split(r'\s+', line.strip())
            if not flds[0].startswith('SS:'):
                continue
            sssp = sss
            sscode_p = sscode
            sss = flds[0][3]
            if sss in ('C','c'):
                sscode = -1
            elif sss == sssp:
                sscode = sscode_p
            else:
                cnt += 1
                sscode = cnt
            ssslist.append(sscode)

    return [erc, ssslist]


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

    bckpred = np.array([MAXDST+1,1]).astype(float) ##distance and its probability for background values


    code, sqn = getSequence(options.sqfile)

    if not code:
        sys.exit('ERROR: Reading sequence failed.')

    if len(sqn) < 1:
        sys.exit('ERROR: Invalid sequence.')

    if options.end < 0:
       options.end = len(sqn)

    if options.beg < 1 or options.end <= options.beg or len(sqn) < options.end:
        sys.exit('ERROR: Invalid sequence start and/or end positions given.')

    ssss = []

    if options.profile:
        code, ssss = readProfileSS(options.profile)
        if not code:
            sys.exit('ERROR: Reading profile failed.')
        if options.beg > 1 and options.end > 1 and len(ssss) != len(sqn):
            sys.exit(
            'ERROR: Inconsistent SS annotation length with the sequence length: ' \
            '%d vs %d.'%(len(ssss),len(sqn)))

    with open(options.output,'w') as ofile:
        lid = 0
        while True:
            if len(npdstpreds) <= lid:
                break
            rid1 = int(npdstpreds[lid,0])
            rid2 = int(npdstpreds[lid,1])
            if len(sqn) < rid1 or rid1 < 1 or len(sqn) < rid2 or rid2 < 1:
                sys.exit('ERROR: Invalid residue number at distance file position %d (%d %d).'%(lid,rid1,rid2))

            ##NOTE: distance = (SUM dst * dprb)/SUM dprb; probability = average
            dst,dprb = (npdstpreds[lid,2]*npdstpreds[lid,3], npdstpreds[lid,3])
            n_dfld = 1
            while True:
                lid += 1
                if len(npdstpreds) <= lid or \
                   int(npdstpreds[lid,0]) != rid1 or int(npdstpreds[lid,1]) != rid2:
                    break
                dst,dprb = (dst + npdstpreds[lid,2]*npdstpreds[lid,3], dprb + npdstpreds[lid,3])
                n_dfld += 1
            if n_dfld < n_dstpreds:
                continue #background: one or more predictions missing
            ##while n_dfld < n_dstpreds:
            ##    dst,dprb = (dst + bckpred[0]*bckpred[1], dprb + bckpred[1]) #this is dummy prediction for background (no contact, or distance>20)
            ##    n_dfld += 1
            if dprb == 0.0:
                continue
            dst /= dprb; dprb /= n_dfld
            ##if MAXDST < dst:
            ##    continue #background

            if dprb < 0.0 or 1.0 < dprb:
                sys.exit('ERROR: Invalid probability value at distance file position %d (%d %d).'%(lid-1,rid1,rid2))
            atm1 = 'CA' if sqn[rid1-1] in ('G','g') else 'CB'
            atm2 = 'CA' if sqn[rid2-1] in ('G','g') else 'CB'
            if rid1 < options.beg or rid2 < options.beg or \
               options.end < rid2 or options.end < rid2:
                continue
            if dprb < options.prb or abs(rid2 - rid1) < options.sep:
                continue
            if options.dst > 0 and float(dst) > options.dst:
                continue
            if len(ssss) and ssss[rid1-1] > 0 and ssss[rid2-1] > 0 and ssss[rid1-1] == ssss[rid2-1]:
                ##pair of positions depends to the same SS element
                continue
            ##reindex:
            rid1 = rid1 - options.beg + 1
            rid2 = rid2 - options.beg + 1
            if options.cstype == 'SIGMOID':
                m = 2.0
                ## plus 4/m to center around prediction
                params = 'SCALARWEIGHTEDFUNC %.3f SUMFUNC 2 SIGMOID %.1f %.1f CONSTANTFUNC -0.5'%(dprb*5.0, dst+4.0/m,m)
            elif options.cstype == 'SIGMOID2':
                m = 2.0
                b = 8.0
                ## minus b/m to make a downhill at dst
                params = 'SCALARWEIGHTEDFUNC %.3f SUMFUNC 2 '%(dprb*5.0) + \
                         'SIGMOID %.1f %.1f SCALARWEIGHTEDFUNC -1.0 SIGMOID %.1f %.1f'%(dst+4.0/m,m, dst+4.0/m-b/m,m)
            elif options.cstype == 'HARMONIC':
                params = '%s %.1f %g'%(options.cstype,dst,1.0-dprb+0.001)
                params = '%s %.1f %g'%(options.cstype,dst,0.5)
            elif options.cstype == 'FLAT_HARMONIC':
                params = '%s %.1f %g %.1f'%(options.cstype,dst,1.0-dprb+0.001,1.0)
                params = '%s %.1f %g %g'%(options.cstype,dst,1.0,4.0)
            elif options.cstype == 'BOUNDED':
                params = '%s %d %.1f %g %.1f TAG'%(options.cstype,0,dst,1.0-dprb+0.001,100.0/(dprb+0.001))
            else:
                sys.exit('ERROR: Unsupported type of Rosetta constraints.')
            ##write distances/constraints
            ofile.write( str('AtomPair ' + 
                atm1 +' '+ str(rid1) +' '+ atm2 +' '+ str(rid2) +' '+ params + "\n"))

#<<>>
