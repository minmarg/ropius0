ROPIUS0, A deep learning-based protocol for protein structure prediction and model selection

(C)2020-2021 Mindaugas Margelevicius,
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
  *  [COMER2](https://github.com/minmarg/comer2)

   The following package required modifications and is included in the 
   software package (Directory `Semantic-Segmentation-Suite`):

  *  [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)

   The directory `pdb70_from_mmcif_200205__selection__promage__SEMSEGM` 
   contains the scheme for the NNs to function properly. This directory 
   also includes the labels (actual distances) of the training and 
   validation datasets.

   The trained NN models are available at:

  *  [ROPIUS0 TensorFlow models](https://zenodo.org/record/4450107/files/ropius0-models.tgz?download=1)

   and should be downloaded and extracted in the root directory of this 
   software package.

# Using ROPIUS0 for predictions

   The description of the two main scripts follows. 
   The first, `srvs/distopred.sh`, is for the prediction of 
   inter-residue distance distributions given a multiple sequence 
   alignment (MSA) only. 
   It produces three predictions made by the three trained 
   instances of the NN model and their final combination: 

```
Usage:
distopred.sh <Options>

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
distopred.sh -i mymsa.afa -o myoutput \
  -C /data/installed-software/comer2 \
  -P /data/installed-software/psipred3.5 \
  -B /data/installed-software/ncbi-blast-2.2.23+
```

   The other script `srvs/rankmodels.sh` is used to rank protein 
   structural models using the ROPIUS0 deep learning framework. 
   It takes an MSA as input for promage (input data structure for the NNs) 
   construction and scores each protein PDB model found in the given 
   directory. 
   On output, it produces the global and residue-wise scores and model 
   ranking in descending order. 
   This script requires the location of a program of the 
   [Rosetta](https://rosettacommons.org/software) software package to be 
   provided:

```
Usage:
rankmodels.sh <Options>

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
rankmodels.sh -i mymsa.afa -d pdbmodels -o myoutput \
  -C /data/installed-software/comer2 \
  -P /data/installed-software/psipred3.5 \
  -B /data/installed-software/ncbi-blast-2.2.23+ \
  -R /data/installed-software/rosetta_bin_linux_2019.35.60890_bundle/main/source/bin/per_residue_energies.static.linuxgccrelease
```

# Docker container

   For facilitating the use of the ROPIUS0 software, a docker image is 
   made available at 
   [https://hub.docker.com/r/minmar/ropius0](https://hub.docker.com/r/minmar/ropius0)
   NN models and external software is included in the ROPIUS0 docker image 
   so that the user can run ROPIUS0 predictions without setting up CUDA 
   drivers and installing external dependencies before. 

   The ROPIUS0 image (2.58 GB, compressed) can be pulled from the Docker 
   repository using the command:

```
docker pull minmar/ropius0
```

   The user's system is supposed to be equipped with a GPU with an 
   installed NVIDIA graphics driver. 
   There is no requirement for CUDA drivers to be installed. 
   However, 
   [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   should be installed to enable the container to connect to the 
   graphics driver. 
   If the user's system already has a CUDA driver installed, it should 
   be at least version 10.1 or upgraded otherwise.

   The container for predicting distances from MSA can be used as follows. 
   Assume first that the user's directory `~/myhostdata` contains an 
   MSA `myhostmsa.afa` in aligned FASTA format. 
   Then predictions can be made and written in subdirectory 
   `ropius0_myhostmsa` of the same directory `~/myhostdata` using the 
   following command:

```
docker run --rm --name=ropius0 --gpus=all --user="$(id -u):$(id -g)" -ti \
   -v ~/myhostdata:/myhome \
   minmar/ropius0 \
   /data/installed-software/ROPIUS0/srvs/distopred.sh \
      -i /myhome/myhostmsa.afa -o /myhome/ropius0_myhostmsa
```

   The first line specifies the container to be removed once finished 
   execution (the image is not removed).
   The host directory `~/myhostdata` is mounted on the container's 
   directory `/myhome` for data exchange in the second line (access 
   from within the container to the host system is not possible).
   The third line specifies the image name. 
   The fourth and the rest of the lines correspond to the container's 
   command (`distopred.sh`) with its arguments. 
   The location of the ROPIUS0 software in the container is 
   `/data/installed-software/ROPIUS0`, and the fourth line specifies the 
   full absolute path to the script. 

   The command for ranking protein structural models that runs in the 
   container has a similar syntax:

```
docker run --rm --name=ropius0 --gpus=all --user="$(id -u):$(id -g)" -ti \
   -v ~/myhostdata:/myhome \
   minmar/ropius0 \
   /data/installed-software/ROPIUS0/srvs/rankmodels.sh \
      -i /myhome/myhostmsa.afa -d /myhome/myhost3Dmodels \
      -o /myhome/ropius0_myhostmsa \
      -c 10
```

   The first three lines are identical to the previous command. 
   The ROPIUS0 script in the fourth line and its arguments below differ. 
   Here, the user's directory of structural models to be evaluated is 
   supposed to be `~/myhostdata/myhost3Dmodels`. 
   Option `-c` specifies the number of CPU cores to use when running 
   Rosetta energy calculation.

   A list of valid command-line options appears upon invoking the 
   executables in the software package with option `-h`.

# References

Margelevicius, M. (2021) ROPIUS0: A deep learning-based protocol for 
protein structure prediction and model selection.

# Funding

The work was supported by the European Regional Development Fund 
[grant No. 01.2.2-LMT-K-718-01-0028]

---

Contact: <mindaugas.margelevicius@bti.vu.lt>

