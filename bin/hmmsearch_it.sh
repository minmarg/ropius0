#!/bin/bash

## Parameters:
INPUT=$1
DB=/data/databases/mgy_clusters_2019_06/mgy_clusters.fa
ROUNDS=4
Eval=1e-3 #1e-4
CPUs=8

## Paths:
SPATH="$(dirname $0)"
HMMERPATH=/data/installed-software/hmmer-3.3-installed/bin

##
[ ! -f ${INPUT} ] && (echo "ERROR: Input file not found."; exit 1)

BNAME=$(basename ${INPUT})
QUERY=${BNAME%.*} #longest name pattern w/o extension
EXT=${BNAME##*.} #shortest matching extension
input=${INPUT}

for ((i=0; i<ROUNDS; i++)); do
    sfx="hmm$((i+1))"
    if [ "${EXT}" == "fa" ]; then
        ## hmmer N1
        ${HMMERPATH}/jackhmmer -o ${QUERY}.${sfx}.out -N 1 -E ${Eval} --domE ${Eval} --cpu ${CPUs} ${INPUT} ${DB}
    elif [ "${EXT}" == "hmm" ]; then
        ## profile search
        ${HMMERPATH}/hmmsearch -o ${QUERY}.${sfx}.out -E ${Eval} --domE ${Eval} --cpu ${CPUs} ${input} ${DB}
    else
        echo "ERROR: Unsupported file extension!"
        exit 1
    fi

    [ $? -eq 0 ] || exit 1

    ## extract pairwise alignments from the output; use a larger value for E-value to include domains that individually score low
    ${SPATH}/hmmer2pwfa.pl -i ${QUERY}.${sfx}.out -o ${QUERY}.${sfx}.pwfa -r 1 -e 0.3

    [ $? -eq 0 ] || exit 1

    ## make STOCKHOLM1 MSA from the pairwise alignments, which will include the aligned query sequence
    ${SPATH}/pwfa2msa.pl -f 2 -i ${QUERY}.${sfx}.pwfa -o ${QUERY}.${sfx}.sto -q ${INPUT} -e 0.3

    [ $? -eq 0 ] || exit 1

    ## make an HMM; NOTE: use --hand to instruct using reference annotation for match states!
    ${HMMERPATH}/hmmbuild --hand ${QUERY}.${sfx}.hmm ${QUERY}.${sfx}.sto

    [ $? -eq 0 ] || exit 1

    ## modify the profile to make match state symbols corresponding to reference states!
    perl -e 'while(<>){s/^(\s+\d+\s+.+\d+\s+)([\w\-]\s)([\w\-]\s)([\w\-]\s[\w\-])$/$1$3$3$4/;print}' ${QUERY}.${sfx}.hmm >${QUERY}.${sfx}.CN.hmm

    [ $? -eq 0 ] || exit 1

    input=${QUERY}.${sfx}.CN.hmm
    EXT=hmm
done

