#!/usr/bin/env python

import sys, os
#import numpy as np
#from scipy.sparse import dok_matrix
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Atom import Atom
from optparse import OptionParser

description = "Get the distance between atoms of the first chain of a PDB file."

def ParseArguments():
    """Parse command-line options.
    """
    parser = OptionParser(description=description)

    parser.add_option("-i", "--infile", dest="input",
                      help="input PDB file", metavar="FILE")
    parser.add_option("-l", "--lower", dest="lower", default=2.0, type=float,
                      help="Distance lower bound; default=%default", metavar="LB")
    parser.add_option("-u", "--upper", dest="upper", default="8.0",
                      help="Distance upper bound; \"inf\" implies printing the lower bound; " \
                        "default=%default", metavar="UB")
    parser.add_option("-r", "--sigfig", dest="rsig", default=3, type=int,
                      help="Distance significant figures (rounding); default=%default", metavar="SF")
    parser.add_option("-t", "--atom", dest="atom", default="CB",
                      help="Type of atoms between which to calculate distance; " \
                        "default=%default", metavar="ATOM_TYPE")
    parser.add_option("-g", "--glyatom", dest="glyatom", default="CA",
                      help="Type of GLY atom for distance calculation; " \
                        "default=%default", metavar="ATOM_TYPE")
    parser.add_option("-c", "--cst", dest="cstype", default="HARMONIC",
                      help="Type of Rosetta constraints (e.g., -c \"SPLINE EPR_DISTANCE\"); " \
                        "default=%default", metavar="TYPE")
    parser.add_option("-p", "--cstpar", dest="cstpar", default="0.5",
                      help="Parameters for Rosetta constraints of the given type " \
                        "(e.g., -p \"1.0 0.5\"); default=%default", metavar="PARAMS")
    parser.add_option("-o", "--outfile", dest="output",
                      help="output file of distances", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.input and options.output):
        sys.stderr.write("ERROR: Input and/or output file is not provided.\n")
        sys.exit()

    if options.lower < 0.0:
        sys.stderr.write("ERROR: Lower bound < 0.\n")
        sys.exit()

    if options.upper != "inf" and int(options.upper) < options.lower:
        sys.stderr.write("ERROR: Upper bound < lower bound.\n")
        sys.exit()

    if options.input and not os.path.isfile(options.input):
        sys.stderr.write("ERROR: Input file does not exist: "+options.input+"\n")
        sys.exit()

    return options


def getAtom(res, atom, glyatom, hetatm=False):
    """Get the name of the atom of interest. atom, atom name; glyatom, atom
    name of the GLY residue; hetatm, consider HETATM records.
    """
    atom = glyatom if res.get_resname() == "GLY" else atom
    if (not hetatm and res.get_id()[0] == " " and res.has_id(atom)) or \
       (hetatm and res.has_id(atom)):
        return [1, res[atom] ]
    return [0, None]


def getDistances(inputfile, lower, upper, atom='CB', glyatom='CA', hetatm=True):
    """Get the distances between the residues of the given chain of pdb
    structrure. inpufile, input pdb file to read; lower, distance lower
    bound; upper, distance upper bound; atom, Type of atom between
    which to calculate distance; nohetatm, glyatom, type of GLY atom for
    distance calculation; hetatm, consider HETATM records.
    Returns a dictionary of distances between residue pairs and a set of 
    residues involved in distance calculation.
    """
    basename = os.path.basename(inputfile)
    name = os.path.splitext(basename)[0]

    pdb = PDBParser().get_structure(name, inputfile)
    model = pdb.get_list()[0]
    chain = model.get_list()[0]
    ress = chain.get_list()

    nres = len(ress)

    #dstmtx = dok_matrix((nres, nres), dtype=np.float32)
    dstmtx = {} # distance matrix/dictionary
    resinc = set() # residue numbers involved in distance calculation

    for n1,r1 in enumerate(ress):
        [found,a1] = getAtom(r1, atom, glyatom, hetatm)
        if not found: continue
        for r2 in ress[n1+1:]:
            [found,a2] = getAtom(r2, atom, glyatom, hetatm)
            if not found: continue
            dst = a2 - a1
            if upper == "inf" and lower <= dst:
                #dstmtx[r1.get_id()[1], r2.get_id()[1]] = lower
                dstmtx[(r1.get_id()[1], r2.get_id()[1])] = lower
            elif lower <= dst and dst <= float(upper):
                #dstmtx[r1.get_id()[1], r2.get_id()[1]] = dst
                dstmtx[(r1.get_id()[1], r2.get_id()[1])] = dst

        resinc.add(r1.get_id()[1])

    return dstmtx, resinc


if __name__ == "__main__":
    options = ParseArguments()
    basename = os.path.basename(options.input)
    name = os.path.splitext(basename)[0]

    pdb = PDBParser().get_structure(name, options.input)
    model = pdb.get_list()[0]
    chain = model.get_list()[0]
    ress = chain.get_list()

    with open(options.output,'w') as ofile:
        for n1,r1 in enumerate(ress):
            [found,a1] = getAtom(r1, options.atom, options.glyatom)
            if not found: continue
            for r2 in ress[n1+1:]:
                [found,a2] = getAtom(r2, options.atom, options.glyatom)
                if not found: continue
                dst = a2 - a1
                if options.upper == "inf" and options.lower <= dst:
                    ofile.write( \
                      str("AtomPair " + \
                      a1.get_name() + " " + str(r1.get_id()[1]) + " " + \
                      a2.get_name() + " " + str(r2.get_id()[1]) + " " + \
                      options.cstype + " " + str(options.lower) + " " + options.cstpar + "\n"))
                elif options.lower <= dst and dst <= int(options.upper):
                    ##write distances/constraints
                    ofile.write( \
                      str("AtomPair " + \
                      a1.get_name() + " " + str(r1.get_id()[1]) + " " + \
                      a2.get_name() + " " + str(r2.get_id()[1]) + " " + \
                      options.cstype + " " + str(round(dst,options.rsig)) + " " + options.cstpar + "\n"))

#<<>>
