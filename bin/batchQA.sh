#!/bin/bash

INSTDIR="$(cd $(dirname "$0")/..; pwd)"

QAPROG=${INSTDIR}/infer/QA4segm_519.py

DATASET=${INSTDIR}/pdb70_from_mmcif_200205__selection__promage__SEMSEGM
MODELPFX=${INSTDIR}/models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN
#MODELPFX=/home2/mindaugas/projects/ROPIUS0/bin/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.001__RUN

#MODELRUN=(0 0  0  1 1  1  2 2)
#MODELEPC=(333 429  499  207 250 133  201 243)
MODELRUN=(2)
MODELEPC=(243)
nmodels=${#MODELEPC[@]}

TARGETDIR=/home/mindaugas/projects/ROPIUS0/QA-tests/FM
TARGETDIR=/home/mindaugas/projects/ROPIUS0/QA-tests/TBM
TARGETDIR=/home/mindaugas/projects/ROPIUS0/QA-tests/FM_ENO1

##maximum number of predicted structures to evaluate
NUM_EVAL=50

##parameters
RANGE=(full)
DSTs=(8 10 20 32 64)
PRBs=(0 0.1 0.2 0.3 0.4)

DSTs=(8 10 20 32)
PRBs=(0 0.1 0.2 0.3)

[ ! -f "${QAPROG}" ] && (echo "ERROR: QA program not found: ${QAPROG}"; exit 1)
[ ! -d "${DATASET}" ] && (echo "ERROR: Dataset directory not found: ${DATASET}"; exit 1)
[ ! -d "${TARGETDIR}" ] && (echo "ERROR: Directory not found: ${TARGETDIR}"; exit 1)

for ((m=0; m<nmodels; m++)); do
    mrun=${MODELRUN[$m]}
    mnum=${MODELEPC[$m]}

    MODEL="${MODELPFX}${mrun}/0${mnum}/model.ckpt"
    MODELNAME="RUN${mrun}e${mnum}"

    ##output evaluation file
    FILE_EVAL="${TARGETDIR}/QA-eval-model${MODELNAME}-$(date +%Y%m%d).txt"

    echo; echo; echo "${FILE_EVAL}"; echo; echo

    evalresults=""
    evalresultsstk=""

    for range in ${RANGE[@]}; do
        for dst in ${DSTs[@]}; do
            for prb in ${PRBs[@]}; do

                namesuffix="QA2__model${MODELNAME}__${range}_d${dst}_p${prb}"

                for subdir in $(ls -1 "${TARGETDIR}"|grep -E '^T'); do
                    target=${subdir}
                    pathname="${TARGETDIR}/${subdir}"
                    maskpathname="${TARGETDIR}/${subdir}/TS--${target}.msk"
                    gt_file="${TARGETDIR}/${subdir}/${target}.txt" ##ground truth

                    [ ! -f "${gt_file}" ] && (echo "ERROR: GT file not found: ${gt_file}"; exit 1)


                    outputpathname="${pathname}/QA2--${target}"
                    [ ! -d "${outputpathname}" ] && (mkdir "${outputpathname}" || exit 1)
                    outputbasename="${target}_${namesuffix}"
                    outputfullfilename="${outputpathname}/${outputbasename}"

                    echo; echo

                    ##structure quality estimation
                    if [ -f "${outputfullfilename}" ]; then
                        echo "I EXISTS: ${outputfullfilename}"
                    else
                        cmd="${QAPROG} --inpmg=${pathname}/${target}.hhb_n6 --inmskdir=${maskpathname} --dataset=${DATASET} "
                        cmd+="--checkpoint_path=${MODEL}  --target=${target} --modelnum=2  --range=${range} --dst=${dst} --prb=${prb}  "
                        cmd+="--out=${outputfullfilename}"

                        echo ${cmd}
                        eval ${cmd}
                        [ $? -eq 0 ] || exit 1
                    fi

                    ##evaluation of results
                    evres=`grep -E "^${target}" "${outputfullfilename}"|sort -r -k2|\
                    perl -e "open(F,'${gt_file}') or die \"\\nERROR: Failed to open ${gt_file}\";
                        while(<F>){\\\$trgs{\\\$2}=\\\$1 if /^\\s*(\\d+)\\s+([^\\-\\s]+)/};close(F);
                        while(<>){
                            @a=split/\\s+/;die \"\\nERROR: \\\$a[0] not found.\\n\" unless exists \\\$trgs{\\\$a[0]};
                            \\\$c++; push @mae, abs(\\\$trgs{\\\$a[0]}-\\\$c);
                            last if $NUM_EVAL<=\\\$c;
                        }
                        die \"\\nERROR: No evaluation points for ${target}\" unless \\\$c;
                        \\\$mean+=\\\$_ foreach @mae; \\\$mean/=\\\$c;
                        \\\$sd+=(\\\$_-\\\$mean)*(\\\$_-\\\$mean) foreach @mae; \\\$sd=sqrt(\\\$sd/\\\$c);
                        printf(\"   %-5s %7.3f %7.3f\",'${target}',\\\$mean,\\\$sd)"`

                    [ $? -eq 0 ] || exit 1

                    evalresults+="${evres}"
                    evalresultsstk+="${evres}\n"
                done

                echo; echo
                echo "Summing and writing evaluation results.."
                evres=`printf "${evalresultsstk}"|\
                  perl -e "while(<>){@a=split/\\s+/;\\\$c++; push @avgs,\\\$a[2];}
                          \\\$mean+=\\\$_ foreach @avgs; \\\$mean/=\\\$c;
                          \\\$sd+=(\\\$_-\\\$mean)*(\\\$_-\\\$mean) foreach @avgs; \\\$sd=sqrt(\\\$sd/\\\$c);
                          printf(\"%-60s %7.3f %7.3f |\",'${namesuffix}',\\\$mean,\\\$sd)"`
                printf "${evres}${evalresults}\n" >>"${FILE_EVAL}"

                evalresults=""
                evalresultsstk=""

            done
        done
    done
done

echo; echo
echo Done.

