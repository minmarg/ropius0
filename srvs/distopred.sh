#!/bin/bash
## (C) 2021 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University
## Predict inter-residue distance distributions using deep learning framework

dirname="$(dirname $0)"
[[ "${dirname:0:1}" != "/" ]] && dirname="$(pwd)/$dirname"
basename="$(basename $0)"

COMER2DIR="/data/installed-software/comer2"
PSIPREDDIR="/data/installed-software/psipred3.5"
BLASTDIR="/data/installed-software/ncbi-blast-2.2.23+"

usage="
Predict inter-residue distance distributions from multiple sequence alignment 
using a deep learning framework.
(C)2021 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University

Usage:
$basename <Options>

Options:

-i <MSA>       Input multiple sequence alignment in aligned FASTA (AFA) format.
               The first sequence in the MSA (gaps removed) corresponds to the 
               target sequence.
               NOTE: HHblits output file, for example, can be converted to AFA 
               format using first the package's utility bin/hhr2pwfa.pl to 
               convert it to intermediate pairwise FASTA format (PWFA) and then
               bin/pwfa2msa.pl to make the final conversion.
-o <out_dir>   Output directory where to place new output files.
               NOTE: If some intermediate files are found to exist, the 
               corresponding stage of the algorithm will be skipped.
       Default=.
-C <path>      COMER2 installation path.
               NOTE: It may be not specified if <out_dir> contains .pro and 
               .cov files constructed using the COMER2 software.
-P <path>      Installation path to PSIPRED.
               NOTE: It may be not specified if <out_dir> contains .pro file
               constructed using the COMER2 software.
-B <path>      Installation path to BLAST+.
               NOTE: It may be not specified if <out_dir> contains .pro file
               constructed using the COMER2 software.
-h             This text.

Example:
${basename} -i mymsa.afa -o myoutput \\
  -C /data/installed-software/comer2 \\
  -P /data/installed-software/psipred3.5 \\
  -B /data/installed-software/ncbi-blast-2.2.23+
"


while getopts "i:o:C:P:B:h" Option
do
    case $Option in
        i ) MSA="${OPTARG}" ;;
        o ) OUTDIR="${OPTARG}" ;;
        C ) COMER2DIR="${OPTARG}" ;;
        P ) PSIPREDDIR="${OPTARG}" ;;
        B ) BLASTDIR="${OPTARG}" ;;
        h ) echo "$usage"; exit 0 ;;
        * ) echo ERROR: Unrecognized argument. >&2; exit 1 ;;
    esac
done
shift $(( $OPTIND - 1 ))


if [ -z "${MSA}" ]; then echo -e "\nERROR: MSA file not specified.\n" >&2; exit 1; fi
if [ ! -f "${MSA}" ]; then echo -e "\nERROR: Input MSA file not found: \"${MSA}\"\n" >&2; exit 1; fi

if [ -z "${OUTDIR}" ]; then OUTDIR=.; fi
if [ ! -d "${OUTDIR}" ]; then 
  mkdir -p "${OUTDIR}" || (echo -e "\nERROR: Failed to create output directory: \"${OUTDIR}\"\n" >&2; exit 1);
fi


combinepredictions="${dirname}/../bin/combinepredictions.py"
modelpreds2CASP="${dirname}/../bin/modelpreds2CASP.py"
pred4segm_519="${dirname}/../infer/pred4segm_519.py"
promage4segm_519="${dirname}/../infer/promage4segm_519.py"

SEMSEGM_DIR="${dirname}/../pdb70_from_mmcif_200205__selection__promage__SEMSEGM"
model_RUN0e333="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN0/0333/model.ckpt"
model_RUN1e207="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN1/0207/model.ckpt"
model_RUN2e201="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN2/0201/model.ckpt"
model_RUN2e243="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN2/0243/model.ckpt"


if [ ! -f "${combinepredictions}" ]; then echo -e "\nERROR: Package program file not found: \"${combinepredictions}\"\n" >&2; exit 1; fi
if [ ! -f "${modelpreds2CASP}" ]; then echo -e "\nERROR: Package program file not found: \"${modelpreds2CASP}\"\n" >&2; exit 1; fi
if [ ! -f "${pred4segm_519}" ]; then echo -e "\nERROR: Package program file not found: \"${pred4segm_519}\"\n" >&2; exit 1; fi
if [ ! -f "${promage4segm_519}" ]; then echo -e "\nERROR: Package program file not found: \"${promage4segm_519}\"\n" >&2; exit 1; fi

if [ ! -d "${SEMSEGM_DIR}" ]; then echo -e "\nERROR: Package directory not found: \"${SEMSEGM_DIR}\"\n" >&2; exit 1; fi
if [ ! -f "${model_RUN0e333}.data-00000-of-00001" ]; then echo -e "\nERROR: Package NN model file not found: \"${model_RUN0e333}\"\n" >&2; exit 1; fi
if [ ! -f "${model_RUN1e207}.data-00000-of-00001" ]; then echo -e "\nERROR: Package NN model file not found: \"${model_RUN1e207}\"\n" >&2; exit 1; fi
if [ ! -f "${model_RUN2e201}.data-00000-of-00001" ]; then echo -e "\nERROR: Package NN model file not found: \"${model_RUN2e201}\"\n" >&2; exit 1; fi
##if [ ! -f "${model_RUN2e243}" ]; then echo -e "\nERROR: Package NN model file not found: \"${model_RUN2e243}\"\n" >&2; exit 1; fi


msaname="$(basename "${MSA}")"
myoutbasename="${msaname}"
profile="${OUTDIR}/${myoutbasename}.pro"
xcovfile="${OUTDIR}/${myoutbasename}.cov"
pmgfullname="${OUTDIR}/${myoutbasename}"
pmgfile="${pmgfullname}.pmg"

if [ ! -f "${pmgfile}" ]; then
    if [[ ! -f "${profile}" || ! -f "${xcovfile}" ]]; then
        ##make profile and xcov file
        if [ -z "${COMER2DIR}" ]; then echo -e "\nERROR: COMER2 location not specified.\n" >&2; exit 1; fi
        if [ ! -d "${COMER2DIR}" ]; then echo -e "\nERROR: COMER2 location not found: \"${COMER2DIR}\"\n" >&2; exit 1; fi

        if [ ! -f "${profile}" ]; then
            ## Construct profile
            makeprosh="${COMER2DIR}/bin/makepro.sh"
            if [ ! -f "${makeprosh}" ]; then echo -e "\nERROR: COMER2 program not found: \"${makeprosh}\"\n" >&2; exit 1; fi

            if [ -z "${PSIPREDDIR}" ]; then echo -e "\nERROR: PSIPRED location not specified.\n" >&2; exit 1; fi
            if [ ! -d "${PSIPREDDIR}" ]; then echo -e "\nERROR: PSIPRED location not found: \"${PSIPREDDIR}\"\n" >&2; exit 1; fi

            if [ -z "${BLASTDIR}" ]; then echo -e "\nERROR: BLAST location not specified.\n" >&2; exit 1; fi
            if [ ! -d "${BLASTDIR}" ]; then echo -e "\nERROR: BLAST location not found: \"${BLASTDIR}\"\n" >&2; exit 1; fi

            echo -e "Constructing COMER2 profile..."
            cmd="\"${makeprosh}\"  -i \"${MSA}\" -o \"${profile}\" -P \"${PSIPREDDIR}\" -B \"${BLASTDIR}\""
            eval "${cmd}"
            if [ $? -ne 0 ]; then
                echo -e "\nERROR: COMER2 profile construction failed. Check the MSA file.\n" >&2
                exit 1
            fi
        fi

        if [ ! -f "${xcovfile}" ]; then
            ## Construct .xcov file
            makecov="${COMER2DIR}/bin/makecov"
            if [ ! -f "${makecov}" ]; then echo -e "\nERROR: COMER2 program not found: \"${makecov}\"\n" >&2; exit 1; fi

            echo -e "Constructing COMER2 cross-covariance file..."
            cmd="\"${makecov}\"  -i \"${MSA}\" -o \"${xcovfile}\" --scale --mi"
            eval "${cmd}"
            if [ $? -ne 0 ]; then
                echo -e "\nERROR: COMER2 cross-covariance file construction failed. Check the MSA file.\n" >&2
                exit 1
            fi
        fi
    fi

    ## Make promage
    echo -e "Making promage..."
    cmd="python3 \"${promage4segm_519}\" --make1=\"${OUTDIR}/${myoutbasename}\""
    eval "${cmd}"
    if [ $? -ne 0 ]; then
        echo -e "\nERROR: Promage construction from profile and xcov file failed.\n" >&2
        exit 1
    fi
fi


if true; then
    ## Run distance predictions
    predR0e333="${OUTDIR}/${myoutbasename}__RUN0e333"
    predR1e207="${OUTDIR}/${myoutbasename}__RUN1e207"
    predR2e201="${OUTDIR}/${myoutbasename}__RUN2e201"

    if [ ! -f "${predR0e333}_pred.prb" ]; then
        echo -e "Predicting distances by NN model 0..."
        cmd="python3 \"${pred4segm_519}\" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN0e333}\""
        cmd+=" --promage=\"${pmgfullname}\" --outpat=\"${predR0e333}\" --model=Encoder-Decoder-Skip"
        eval "${cmd}"
        if [ $? -ne 0 ]; then
            echo -e "\nERROR: Prediction using model 0 failed.\n" >&2
            exit 1
        fi
    fi

    if [ ! -f "${predR1e207}_pred.prb" ]; then
        echo -e "Predicting distances by NN model 1..."
        cmd="python3 \"${pred4segm_519}\" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN1e207}\""
        cmd+=" --promage=\"${pmgfullname}\" --outpat=\"${predR1e207}\" --model=Encoder-Decoder-Skip"
        eval "${cmd}"
        if [ $? -ne 0 ]; then
            echo -e "\nERROR: Prediction using model 1 failed.\n" >&2
            exit 1
        fi
    fi

    if [ ! -f "${predR2e201}_pred.prb" ]; then
        echo -e "Predicting distances by NN model 2..."
        cmd="python3 \"${pred4segm_519}\" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN2e201}\""
        cmd+=" --promage=\"${pmgfullname}\" --outpat=\"${predR2e201}\" --model=Encoder-Decoder-Skip"
        eval "${cmd}"
        if [ $? -ne 0 ]; then
            echo -e "\nERROR: Prediction using model 2 failed.\n" >&2
            exit 1
        fi
    fi
fi


## Combine all predictions
outputprbfile="${OUTDIR}/${myoutbasename}__pred.prb"
TARGET=MYTARGET

echo -e "Combining predictions..."
cmd="python3 \"${modelpreds2CASP}\" --infile=\"${predR0e333}_pred.prb,${predR1e207}_pred.prb,${predR2e201}_pred.prb\""
cmd+=" --target=${TARGET} --outfile=\"${outputprbfile}\""
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Combination of predictions failed.\n" >&2
    exit 1
fi

nonavgprbfile="${OUTDIR}/${myoutbasename}__pred__nonavg.prb"

echo -e "Concatenating predictions..."
cmd="python3 \"${combinepredictions}\" --infile=\"${predR0e333}_pred.prb,${predR1e207}_pred.prb,${predR2e201}_pred.prb\""
cmd+=" --output=\"${nonavgprbfile}\""
msgstr="Non-averaged probabilities: ${nonavgprbfile}"
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Concatenation of predictions failed.\n" >&2
    msgstr=""
    ##exit 1
fi


echo -e "\n\nFinished. Output file: ${outputprbfile}\n${msgstr}\n"
exit 0

