# JetML

## Paper
This repository contains codes and data from [arXiv:2206.01628](https://arxiv.org/abs/2206.01628). Plotting macros are avaialble in ```macros``` folder.

## Usage

### 1. Clone this repo

```
git clone https://github.com/ustcllh/JetML.git
cd JetML
```

Note: 

```git lfs``` is used for tracking files in ```data/Classified``` folder.

Link: [Git LFS](https://git-lfs.github.com)

Full dataset is available from [google drive](https://drive.google.com/drive/folders/12hxxRplZKlBS7Ds_kRvIXiyRumq2MsgZ?usp=sharing).



### 2. Virtual Machine
All dependencies including PyTorch and ROOT are installed in a pre-confiured virtual machine. One can access the container with either ```docker``` or ```singularity``` depending on platforms. The pre-configured docker container is GPU-enabled. You may need ```nvidia-docker``` to make use of a Nvidia GPU. In addition, you may want to increase container memory size from docker preferences.

For AWS users, simply launch a p2 instance with Deep Learning AMI which comes with ```nvidia-docker```.

#### Option 1:  Docker
Sinmply pull a pre-configured docker image and run.
```
# start docker container
# mount current directory with -v argument
docker run -it -v$PWD:/workspace/JetML ustcllh/jetml:gpu_latest

# Optional: start nvidia-docker container
docker run -it --gpus all -v$PWD:/workspace/JetML ustcllh/jetml:gpu_latest

# compile source code
cd /workspace/JetML
make

# install python3 packages
pip install -r requirements.txt
```

You may want to check GPU usage with the following command.

```
nvidia-smi
```

#### Option 2: Singularity
For people who are running on shared computing clusters, ```singularity``` is an alternative to docker. Simply pull the container.
```
singularity pull docker://ustcllh/jetml:gpu_latest
```
An executable with extension ".sif" can be used either interactively or in batch mode.

To use the container interactively:
```
# access the container shell
./jetml_gpu_latest.sif

# current directory is automatically mounted
# compile source code
make

# install python3 packages
pip install -r requirements.txt

# exit from the container
exit
```

In most of cases one may want to use it in batch mode:
```
# without nvidia runtime environment
singularity exec jetml_gpu_latest.sif <commands>

# with nvidia runtime environment
singularity exec --nv jetml_gpu_latest.sif <commands>
```

### 3. Download dataset

#### pre-processed data

Full dataset is available from [google drive](https://drive.google.com/drive/folders/12hxxRplZKlBS7Ds_kRvIXiyRumq2MsgZ?usp=sharing).

Root files in ```Training``` and ```Validation``` folders can be used directly in the training of a neural network. Data in ```Classified``` folder contains output from a pre-trained neural network.


#### Monte-carlo simulation

Simulated events with event genetators such as Jewel and Pythia are provided. In this study a human-readable format ```.pu14``` is used. For conversion from ```.hepmc``` to ```.pu14```, tools are provided in another repo [ustcllh/hep_converter](https://github.com/ustcllh/hep_converter).
```
# Jewel with Recoil
https://jetquenchingtools.web.cern.ch/JetQuenchingTools/samples/jewel_R_2.2_5.02_Sep18/

# Pythia 8
https://jetquenchingtools.web.cern.ch/JetQuenchingTools/samples/pythia8/

# Thermal Background
https://jetquenchingtools.web.cern.ch/JetQuenchingTools/samples/thermal/
```

### 4. Feature Engineering

This procedure contains the following steps:
- Mixing of one hard event and one thermal event
- Background Subtraction
- Jet Finding
- Jet De-clustering
- Access to jet substructure variables

```
# start feature engineering
./doStructure.sh -i <path/to/hard_events.pu14> -b <path/to/thermal_events.pu14> -o <path/to/output_file.root> -n <no. of events>
```

### 5. Train a neural network

By default, datasets downloaded from section 3 are used to train a neural network built on top of LSTM layers. Increase the hyper-tunner parameter ```max_evals``` for more trials. A pre-trianed neural network is accessible in ```model``` folder with a json file containing hyper-parameters.

```
# start training
./doTraining.sh
```

### 6. Use the trained neural network

A trained neural network can be used as a classifier.
```
# usage
# ./doClassify.sh -i <path/to/input_file.root> -o <path/to/output_file.root> -t csejet -c <path/to/classifier_file(without extension)>

# example
./doClassify.sh -i ./data/Validation/jewel_R_csejet_pt200_rg0p1_mult7000.root -o jewel_R_csejet_pt200_rg0p1_mult7000_classified.root -t csejet -c ./model/zcut0p1_beta0_csejet_mse_mult7000_pt200_best
```