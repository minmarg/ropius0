#!/usr/bin/env python

import sys, os, re
import math
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.PDBIO import Select
from optparse import OptionParser

description = "Translate a Rosetta model to the full-featured CASP 3D model."
chainNA = "n/a"
MAXERR = 8.0

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="input PDB file", metavar="FILE")
    parser.add_option("-s", "--scfile", dest="scores",
                      help="input Rosetta per-residue scores file", metavar="FILE")
    parser.add_option("-c", "--chain", dest="cid", default=chainNA,
                      help="PDB chain id to extract from file; " \
                          "by default, the first chain is selected", metavar="ID")
    parser.add_option("-a", "--author", type=int, dest="author", default=1,
                      help="Author number; can be 1, primary, or 2, secondary submitter",
                      metavar="NUMBER")
    parser.add_option("-t", "--target", dest="target",
                      help="Target name", metavar="TARGET")
    parser.add_option("-m", "--model", type=int, dest="model",
                      help="Model number", metavar="MODEL")
    parser.add_option("--parent", dest="parent",
                      help="Parent structure ID(s) to be written in the resulting file",
                      metavar="ID(s)")
    parser.add_option("--write_header", action="store_true",
                      help="Write header information in CASP format at the beginning of file")
    parser.add_option("--write_end", action="store_true",
                      help="Write END at the end of file")
    parser.add_option("-o", "--outfile", dest="output",
                      help="output PDB file", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.input and options.scores):
        sys.stderr.write("ERROR: Not all input files are provided.\n")
        sys.exit()

    if not options.output:
        sys.stderr.write("ERROR: Output filename is not provided.\n")
        sys.exit()

    if 4 < len(options.cid) or len(options.cid) < 1:
        sys.stderr.write("ERROR: Invalid chain id specified.\n")
        sys.exit()

    if not (options.author and (options.author == 1 or options.author == 2)):
        sys.stderr.write("ERROR: Author number is not provided or invalid.\n")
        sys.exit()

    if not (options.target and options.model):
        sys.stderr.write("ERROR: Target name and/or model number is not provided.\n")
        sys.exit()

    if options.model < 1:
        sys.stderr.write("ERROR: Invalid model number given: "+str(options.model)+"\n")
        sys.exit()

    if options.input and not os.path.isfile(options.input):
        sys.stderr.write("ERROR: Input PDB file does not exist: "+options.input+"\n")
        sys.exit()

    if options.scores and not os.path.isfile(options.scores):
        sys.stderr.write("ERROR: Input scores file does not exist: "+options.scores+"\n")
        sys.exit()

    return options



class ChainSelect(Select):
    def __init__(self, cid):
        self.cid = cid

    def accept_chain(self, chain):
        if chain.get_id() == self.cid:
            return 1
        else:
            return 0


def GetPositionalScores(filename):
    """Read per-residue scores/energies from the given file.
    """
    mydefname = 'GetPositionalScores: '
    if not os.path.isfile(filename):
        sys.exit('ERROR: '+mydefname+'Scores file not found: '+ filename)

    errorlist = []
    totalscore = 0
    erc = 1
    pos = 0

    with open(filename) as fp:
        line = fp.readline()
        for line in fp:
            flds = re.split(r'\s+', line.strip())
            pos += 1
            nflds = len(flds)
            if nflds < 3+2:
                sys.stderr.write('ERROR: Unexpected number of fields at scores file position %d\n'%(pos))
                erc = 0
                break
            score = sum(np.array(flds[3:nflds-2]).astype(float))
            totalscore += score
            e = math.exp(score) if score < math.log(MAXERR) else MAXERR
            errorlist.append(e)

    sys.stderr.write("Total model score, %.3f\n"%(totalscore))
    return erc, errorlist



if __name__ == "__main__":
    options = ParseArguments()
    basename = os.path.basename(options.input)
    name, extension = os.path.splitext(basename)

    code, errorlst = GetPositionalScores(options.scores)

    if not code:
        sys.exit('ERROR: Reading model per-residue scores failed.')

    if extension == '.cif':
        pdb = MMCIFParser().get_structure(name, options.input)
    else:
        pdb = PDBParser().get_structure(name, options.input)

    model = pdb.get_list()[0]

    if options.cid == chainNA:
        ##get the first chain if not specified
        chain = model.get_list()[0]
        options.cid = chain.get_id()
    else:
        chain = model[options.cid]

    if 1 < len(options.cid):
        newid = options.cid[0]
        if model.has_id(newid):
            model.detach_child(newid)
        chain.id = newid

    ress = chain.get_list()
    nres = len(ress)

    if nres != len(errorlst):
        sys.exit('ERROR: Inconsistent structure lengths: %d vs %d.'%(nres, len(errorlst)))

    for n1,r1 in enumerate(ress):
        #atms = r1.get_list()
        for a1 in r1:
            a1.set_bfactor(errorlst[n1])

    with open(options.output,'w') as ofile:
        if options.write_header:
            ofile.write('PFRMAT TS\n')
            ofile.write('TARGET %s\n'%(options.target))
            if options.author == 1:
                ofile.write('AUTHOR 1929-2214-0552\n') ##ROPIUS0 registration code in CASP14
            elif options.author == 2:
                ofile.write('AUTHOR 1137-7210-5150\n') ##ROPIUS0QA registration code in CASP14
            else:
                sys.exit('ERROR: Invalid Author number: %d.'%(options.author))
            ofile.write('METHOD Restraint-Oriented Protocol for Inference and \n')
            ofile.write('METHOD Understanding of protein Structures.\n')
            ofile.write('METHOD Based on COMER, Rosetta, and deep learning.\n')

        ofile.write('MODEL  %d\n'%(options.model))

        if options.parent:
            ofile.write('PARENT %s\n'%(options.parent))

        io = PDBIO()
        io.set_structure(pdb) #chain)
        io.save(ofile, ChainSelect(chain.get_id()), write_end=options.write_end)

#<<>>

