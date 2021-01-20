ROPIUS0, A deep learning-based protocol for protein structure prediction and model selection

(C)2020 Mindaugas Margelevicius,
Institute of Biotechnology, Vilnius University

# Description

   ROPIUS0's approach to protein structure prediction is based on homology 
   modeling when structural templates can be identified. It aims to select 
   the most accurate models from a set of independently generated 
   structural models otherwise. The selection of models obtained by 
   either approach relies on the match between predicted inter-residue 
   distances and those observed in the model. Differences in distances 
   determine the estimates of model quality. Three independently trained 
   encoder-decoder convolutional neural networks for producing distance 
   predictions are at the core of the ROPIUS0 protocol.

   ROPIUS0 is licensed under GNU General Public License version 3. 
   Please find the LICENSE and COPYING files.

# Software Requirements

   ROPIUS0 is mainly written in Python and Perl and, therefore, does not 
   require installation. However, different parts of the ROPIUS0 protocol 
   depend on external software packages. The neural network (NN) part is 
   implemented in TensorFlow v1.15 and dependent upon associated python 
   packages. Other dependencies include:
 
  *  [BioPython](https://biopython.org)
  *  [Mask RCNN](https://github.com/matterport/Mask_RCNN)
  *  [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)

   Since the latter required modifications, it is included in the present 
   software package (Directory `Semantic-Segmentation-Suite`). Also, the 
   directory `pdb70_from_mmcif_200205__selection__promage__SEMSEGM` 
   contains the scheme for the NNs to function properly. This directory
   also includes the labels (actual distances) of the training and 
   validation datasets. The references to these directories in the 
   scripts should be adjusted accordingly. 

   The trained NN models are available at:

  *  [ROPIUS0 TensorFlow models](https://zenodo.org/record/4450107/files/ropius0-models.tgz?download=1)

   and should be downloaded and extracted in the root directory of this 
   software package.

# Prediction example

   The following shell script snippet demonstrates distance prediction 
   for target protein `5M33_B` by the three trained instances of the NN 
   model. It is assumed that the promage (input data structure for the 
   NNs) for the sequence of `5M33_B` has been constructed using 
   `infer/promage4cother_519.py`

```
TARGET=5M33_B

PREDprog="infer/pred4segm_519.py"

PROMAGE_DIR="my_promages"
SEMSEGM_DIR="pdb70_from_mmcif_200205__selection__promage__SEMSEGM"
model_RUN0e333="models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN0/0333/model.ckpt"
model_RUN1e207="models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN1/0207/model.ckpt"
model_RUN2e201="models/checkpoints_519_normonly_from_scratch__Encoder-Decoder-Skip_lr0.0001__RUN2/0201/model.ckpt"

## Prediction of distances with maximum probability
## using the first instance of the NN model: RUN0e333
python3 "${PREDprog}" --dataset="${SEMSEGM_DIR}" --checkpoint_path="${model_RUN0e333}" --promage="${PROMAGE_DIR}/${TARGET}" --model=Encoder-Decoder-Skip

mv "${TARGET}_pred.prb" "${TARGET}_pred__RUN0e333.prb"
mv "${TARGET}_pred.png" "${TARGET}_pred__RUN0e333.png"

## using the second instance of the NN model: RUN1e207
python3 "${PREDprog}" --dataset="${SEMSEGM_DIR}" --checkpoint_path="${model_RUN1e207}" --promage="${PROMAGE_DIR}/${TARGET}" --model=Encoder-Decoder-Skip

mv "${TARGET}_pred.prb" "${TARGET}_pred__RUN1e207.prb"
mv "${TARGET}_pred.png" "${TARGET}_pred__RUN1e207.png"

## using the third instance of the NN model: RUN2e201
python3 "${PREDprog}" --dataset="${SEMSEGM_DIR}" --checkpoint_path="${model_RUN2e201}" --promage="${PROMAGE_DIR}/${TARGET}" --model=Encoder-Decoder-Skip

mv "${TARGET}_pred.prb" "${TARGET}_pred__RUN2e201.prb"
mv "${TARGET}_pred.png" "${TARGET}_pred__RUN2e201.png"

```

   The three predictions can be further used independently or combined 
   using one of the programs, e.g.,

```
   python3 bin/modelpreds2CASP.py --infile=${TARGET}_pred__RUN0e333.prb,${TARGET}_pred__RUN1e207.prb,${TARGET}_pred__RUN2e201.prb --target=${TARGET} --outfile=my_final_preds.out
```

# References

Margelevicius, M. (2021) ROPIUS0: A deep learning-based protocol for 
protein structure prediction and model selection.

# Funding

The work was supported by the European Regional Development Fund 
[grant No. 01.2.2-LMT-K-718-01-0028]

---

Contact: <mindaugas.margelevicius@bti.vu.lt>

