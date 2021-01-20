#!/usr/bin/env python

import sys, os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.PDBIO import Select
from optparse import OptionParser

description = "Remove a number of residues from the beginning and/or end of the " \
"structure chain."
chainNA = "n/a"

parser = OptionParser(description=description)

parser.add_option("-i", "--infile", dest="input",
                  help="input PDB file", metavar="FILE")
parser.add_option("-c", "--chain", dest="cid", default=chainNA,
                  help="PDB chain id to extract from file; " \
                      "by default, the first chain is selected", metavar="ID")
parser.add_option("-b", "--nbres", dest="nbres", default=0, type=int,
                  help="Number of residues to remove from the beginning; " \
                      "default=%default", metavar="NUM")
parser.add_option("-e", "--neres", dest="neres", default=0, type=int,
                  help="Number of residues to remove from the end; " \
                      "default=%default", metavar="NUM")
parser.add_option("-o", "--outfile", dest="output",
                  help="output PDB file", metavar="FILE")

(options, args) = parser.parse_args()

if not (options.input and options.output):
    sys.stderr.write("ERROR: Input and/or output file is not provided.\n")
    sys.exit()

if 4 < len(options.cid) or len(options.cid) < 1:
    sys.stderr.write("ERROR: Invalid chain id specified.\n")
    sys.exit()

if options.input and not os.path.isfile(options.input):
    sys.stderr.write("ERROR: Input file does not exist: "+options.input+"\n")
    sys.exit()

if options.nbres < 0 or options.neres < 0:
    sys.stderr.write("ERROR: Invalid numbers of residues: *"+ \
        str(options.nbres)+","+str(options.neres)+")\n")
    sys.exit()


class ChainSelect(Select):
    def __init__(self, cid, res1, resl):
        self.cid = cid
        self.res1 = res1
        self.resl = resl

    def accept_chain(self, chain):
        if chain.get_id() == self.cid:
            return 1
        else:
            return 0
    def accept_residue(self, res):
        nfrombeg = res.get_id()[1] - self.res1.get_id()[1]
        ntoend = self.resl.get_id()[1] - res.get_id()[1]
        if nfrombeg < options.nbres or ntoend < options.neres:
            return 0
        return 1


basename = os.path.basename(options.input)
name, extension = os.path.splitext(basename)

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

res1 = chain.get_list()[0]
resl = chain.get_list()[-1]

io = PDBIO()
io.set_structure(pdb) #chain)
io.save(options.output, ChainSelect(chain.get_id(), res1, resl))

