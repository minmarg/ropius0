#!/bin/bash
## (C) 2021 Mindaugas Margelevicius, Institute of Biotechnology, Vilnius University
## Rank protein structural models using deep learning framework

dirname="$(dirname $0)"
[[ "${dirname:0:1}" != "/" ]] && dirname="$(pwd)/$dirname"
basename="$(basename $0)"

COMER2DIR="/data/installed-software/comer2"
PSIPREDDIR="/data/installed-software/psipred3.5"
BLASTDIR="/data/installed-software/ncbi-blast-2.2.23+"
per_residue_energies="/data/installed-software/rosetta_bin_linux_2019.35.60890_bundle/main/source/bin/per_residue_energies.static.linuxgccrelease"

usage="
Rank protein structural models using a deep learning framework.
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
-d <directory> Input directory of structural (pdb) models to rank, generated
               for the target sequence.
-o <out_dir>   Output directory where to place new output files.
               NOTE: If some intermediate files are found to exist, the 
               corresponding stage of the algorithm will be skipped.
       Default=.
-c <n_cpus>    Number of CPU cores to use.
       Default=1
-C <path>      COMER2 installation path.
               NOTE: It may be not specified if <out_dir> contains .pro and 
               .cov files constructed using the COMER2 software.
-P <path>      Installation path to PSIPRED.
               NOTE: It may be not specified if <out_dir> contains .pro file
               constructed using the COMER2 software.
-B <path>      Installation path to BLAST+.
               NOTE: It may be not specified if <out_dir> contains .pro file
               constructed using the COMER2 software.
-R <program_file> The per_residue_energies program of the Rosetta software
               package.
-h             This text.

Example:
${basename} -i mymsa.afa -d pdbmodels -o myoutput \\
  -C /data/installed-software/comer2 \\
  -P /data/installed-software/psipred3.5 \\
  -B /data/installed-software/ncbi-blast-2.2.23+ \\
  -R /data/installed-software/rosetta_bin_linux_2019.35.60890_bundle/main/source/bin/per_residue_energies.static.linuxgccrelease
"
##-S <program_file> The score_jd2 program of the Rosetta software package.
##-S /data/installed-software/rosetta_bin_linux_2019.35.60890_bundle/main/source/bin/score_jd2.static.linuxgccrelease \\


while getopts "i:d:o:c:C:P:B:S:R:h" Option
do
    case $Option in
        i ) MSA="${OPTARG}" ;;
        d ) INDIR="${OPTARG}" ;;
        o ) OUTDIR="${OPTARG}" ;;
        c ) nCPUs="${OPTARG}" ;;
        C ) COMER2DIR="${OPTARG}" ;;
        P ) PSIPREDDIR="${OPTARG}" ;;
        B ) BLASTDIR="${OPTARG}" ;;
##        S ) score_jd2="${OPTARG}" ;;
        R ) per_residue_energies="${OPTARG}" ;;
        h ) echo "$usage"; exit 0 ;;
        * ) echo ERROR: Unrecognized argument. >&2; exit 1 ;;
    esac
done
shift $(( $OPTIND - 1 ))


if [ -z "${MSA}" ]; then echo -e "\nERROR: MSA file not specified.\n" >&2; exit 1; fi
if [ ! -f "${MSA}" ]; then echo -e "\nERROR: Input MSA file not found: \"${MSA}\"\n" >&2; exit 1; fi

if [ -z "${INDIR}" ]; then echo -e "\nERROR: Input directory of models not specified.\n" >&2; exit 1; fi
if [ ! -d "${INDIR}" ]; then echo -e "\nERROR: Input directory not found: \"${INDIR}\"\n" >&2; exit 1; fi

if [ -z "${OUTDIR}" ]; then OUTDIR=.; fi
if [ ! -d "${OUTDIR}" ]; then 
  mkdir -p "${OUTDIR}" || (echo -e "\nERROR: Failed to create output directory: \"${OUTDIR}\"\n" >&2; exit 1);
fi

if [ -z "${nCPUs}" ]; then nCPUs=1; fi
if [[ !( "${nCPUs}" =~ ^[1-9][0-9]*$ ) ]]; then
  echo -e "\nERROR: Invalid number of CPU cores specified: \"${nCPUs}\"\n" >&2
  exit 1
fi


##if [ -z "${score_jd2}" ]; then echo -e "\nERROR: Program score_jd2 not specified.\n" >&2; exit 1; fi
##if [ ! -f "${score_jd2}" ]; then echo -e "\nERROR: Program file score_jd2 not found: \"${score_jd2}\"\n" >&2; exit 1; fi

if [ -z "${per_residue_energies}" ]; then echo -e "\nERROR: Program per_residue_energies not specified.\n" >&2; exit 1; fi
if [ ! -f "${per_residue_energies}" ]; then echo -e "\nERROR: Program file per_residue_energies not found: \"${per_residue_energies}\"\n" >&2; exit 1; fi


pred4segm_519="${dirname}/../infer/pred4segm_519.py"
promage4segm_519="${dirname}/../infer/promage4segm_519.py"
QA4segm_519="${dirname}/../infer/QA4segm_519.py"
QAcombine_plus_RosettaE="${dirname}/../bin/QAcombine_plus_RosettaE.pl"

SEMSEGM_DIR="${dirname}/../pdb70_from_mmcif_200205__selection__promage__SEMSEGM"
model_RUN0e333="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN0/0333/model.ckpt"
model_RUN1e207="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN1/0207/model.ckpt"
model_RUN2e201="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN2/0201/model.ckpt"
model_RUN2e243="${dirname}/../models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN2/0243/model.ckpt"


if [ ! -f "${pred4segm_519}" ]; then echo -e "\nERROR: Package program file not found: \"${pred4segm_519}\"\n" >&2; exit 1; fi
if [ ! -f "${promage4segm_519}" ]; then echo -e "\nERROR: Package program file not found: \"${promage4segm_519}\"\n" >&2; exit 1; fi
if [ ! -f "${QA4segm_519}" ]; then echo -e "\nERROR: Package program file not found: \"${QA4segm_519}\"\n" >&2; exit 1; fi
if [ ! -f "${QAcombine_plus_RosettaE}" ]; then echo -e "\nERROR: Package program file not found: \"${QAcombine_plus_RosettaE}\"\n" >&2; exit 1; fi

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


if false; then
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


## Make link to input directory
indirfull="$(cd "${INDIR}"; pwd)"
indirname="$(basename "${INDIR}")"
if [ "${indirname:0:1}" == "." ]; then
    indirname=input
fi
modelsdir="${OUTDIR}/${indirname}"
if [ -d "${modelsdir}" ]; then
    echo -e "\n\nWARNING: Directory \"${modelsdir}\" exists! Assuming it is the input directory of pdb models.\n" >&2
    sleep 3
else
    echo -e "Creating a link to pdb models directory..."
    cmd="ln -s \"${indirfull}\" \"${modelsdir}\""
    eval "${cmd}"
    if [ $? -ne 0 ]; then
        echo -e "\nERROR: Failed to create a link to the input directory.\n" >&2
        exit 1
    fi
fi

## Make masks
modelsmskdir="${modelsdir}.msk"
echo -e "Making masks for the pdb models..."
cmd="python3 \"${promage4segm_519}\" --make1=\"${pmgfullname}\" --makemasks=\"${modelsdir}\""
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Making masks failed.\n" >&2
    exit 1
fi


## Calculate Rosetta per-residue energy scores
modelsrscdir="${modelsdir}.rsc"
if [ -d "${modelsrscdir}" ]; then
    rm -R "${modelsrscdir}" || (echo -e "\nERROR: Failed to remove directory: \"${modelsrscdir}\"\n" >&2; exit 1);
fi
mkdir "${modelsrscdir}" || (echo -e "\nERROR: Failed to create directory: \"${modelsrscdir}\"\n" >&2; exit 1);

echo -e "Calculating Rosetta per-residue energy scores..."
ls -1 "${modelsdir}"/* | xargs -i -P ${nCPUs} sh -c "name=\$(basename {}); \"${per_residue_energies}\" -in:file:s {}  -score:weights talaris2014_cst.wts -restore_talaris_behavior -out:file:silent \"${modelsrscdir}/\${name}.rsc\""
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Rosetta scores calculation failed.\n" >&2
    exit 1
fi


## Calculate the match between the prediction and the pdb models
matchR0e333="${OUTDIR}/${indirname}__QA2__modelRUN0e333__full_d20_p0.3"
matchR1e207="${OUTDIR}/${indirname}__QA2__modelRUN1e207__full_d10_p0.1"
matchR2e201="${OUTDIR}/${indirname}__QA2__modelRUN2e201__full_d8_p0.1"
MODELNUM=7
TARGET=MYTARGET

echo -e "Calculating the match between the pdb models and prediction 0..."
cmd="python3 \"${QA4segm_519}\" --inpmg=\"${pmgfullname}\" --inmskdir=\"${modelsmskdir}\""
cmd+=" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN0e333}\" --range=full"
cmd+=" --target=${TARGET} --modelnum=${MODELNUM} --dst=20 --prb=0.3 --out=\"${matchR0e333}\""
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Match calculation using NN model 0 failed.\n" >&2
    exit 1
fi

echo -e "Calculating the match between the pdb models and prediction 1..."
cmd="python3 \"${QA4segm_519}\" --inpmg=\"${pmgfullname}\" --inmskdir=\"${modelsmskdir}\""
cmd+=" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN1e207}\" --range=full"
cmd+=" --target=${TARGET} --modelnum=${MODELNUM} --dst=10 --prb=0.1 --out=\"${matchR1e207}\""
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Match calculation using NN model 1 failed.\n" >&2
    exit 1
fi

echo -e "Calculating the match between the pdb models and prediction 2..."
cmd="python3 \"${QA4segm_519}\" --inpmg=\"${pmgfullname}\" --inmskdir=\"${modelsmskdir}\""
cmd+=" --dataset=\"${SEMSEGM_DIR}\" --checkpoint_path=\"${model_RUN2e201}\" --range=full"
cmd+=" --target=${TARGET} --modelnum=${MODELNUM} --dst=8 --prb=0.1 --out=\"${matchR2e201}\""
eval "${cmd}" 
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Match calculation using NN model 2 failed.\n" >&2
    exit 1

fi


## Combine all predictions with Rosetta energy terms
outputrankfile="${OUTDIR}/${indirname}__QA2.rank"

echo -e "Combining predictions with Rosetta energy terms..."
if [ -f "${outputrankfile}" ]; then rm "${outputrankfile}"; fi
cmd="\"${QAcombine_plus_RosettaE}\" -o \"${outputrankfile}\" -d \"${modelsrscdir}\""
cmd+=" -t 7,10,12,22 -w 0.2 \"${matchR2e201}\" \"${matchR1e207}\" \"${matchR0e333}\""
eval "${cmd}"
if [ $? -ne 0 ]; then
    echo -e "\nERROR: Combination of predictions failed.\n" >&2
    exit 1
fi

outputsrtrankfile="${OUTDIR}/${indirname}__QA2__sorted.rank"
perl -e 'while(<>){
    next unless/^\S+\s+[\d\.]+\s+(?:[\d\.]+|X)\s+(?:[\d\.]+|X)\s/;
    @a=split/\s+/;
    $H{$_}=$a[1]
  }
  @s=sort{$H{$b}<=>$H{$a}} keys %H;
  print foreach @s' "${outputrankfile}" >"${outputsrtrankfile}"


## Remove created link and exit
rm "${modelsdir}"

echo -e "\n\nFinished. Output file: ${outputrankfile}\n"
exit 0

