#!/usr/bin/env python

import sys, os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.SeqUtils import seq1
from Bio.PDB.PDBIO import Select
from optparse import OptionParser


description = "Write one-letter residue sequence in FASTA."
chainNA = "n/a"


def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="Input PDB file", metavar="FILE")
    parser.add_option("-c", "--chain", dest="cid", default=chainNA,
                      help="PDB chain id to extract from file; " \
                        "by default, the first chain is selected", metavar="ID")
    parser.add_option("--nohetatm", action="store_true", dest="nohetatm",
                      help="Ignore HETATM records")
    parser.add_option("-n", "--name", dest="name",
                      help="Name of the sequence", metavar="NAME")
    parser.add_option("-o", "--outfile", dest="output",
                      help="Output PDB file", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.input):
        sys.stderr.write("ERROR: Input file is not provided.\n")
        sys.exit()

    if not options.name:
        sys.stderr.write("ERROR: Name should be provided.\n")
        sys.exit()

    if 4 < len(options.cid) or len(options.cid) < 1:
        sys.stderr.write("ERROR: Invalid chain id specified.\n")
        sys.exit()

    if options.input and not os.path.isfile(options.input):
        sys.stderr.write("ERROR: Input file does not exist: "+options.input+"\n")
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

    def accept_residue(self, res):
        return res.get_id()[0] == " "


def getChainSequence(inputfile, cid=chainNA, nohetatm=True):
    """Get the on-letter sequence of the given chain of pdb
    structrure. inpufile, input pdb file to read; cid, chain 
    identifier; nohetatm, flag commanding not to process HETATM
    records.
    """
    basename = os.path.basename(inputfile)
    name, extension = os.path.splitext(basename)

    if extension == '.cif':
        pdb = MMCIFParser().get_structure(name, inputfile)
    else:
        pdb = PDBParser().get_structure(name, inputfile)

    model = pdb.get_list()[0]

    if cid == chainNA:
        ##get the first chain if not specified
        chain = model.get_list()[0]
        cid = chain.get_id()
    else:
        chain = model[cid]

    if 1 < len(cid):
        newid = cid[0]
        if model.has_id(newid):
            model.detach_child(newid)
        chain.id = newid

    seq = "" # sequence
    num = [] # residue numbers for the sequence

    for res in chain.get_residues():
        if not nohetatm or \
           (nohetatm and res.get_id()[0] == " "):
            seq += seq1(res.get_resname());
            num.append(res.get_id()[1]);

    return {'seq': seq, 'num': num}


if __name__ == "__main__":
    options = ParseArguments()
    pdbdct = getChainSequence(options.input, options.cid, options.nohetatm)
    if options.output:
        with open(options.output,'w') as ofile:
            ofile.write(">" + options.name + "\n" + pdbdct['seq'])
    else:
        print(">" + options.name + "\n" + pdbdct['seq'])

#<<>>
