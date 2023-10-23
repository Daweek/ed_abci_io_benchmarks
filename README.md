# A couple of benchmarks to measure I/O on synthetic datasets (FractalDB) using ABCI super AI system.

Author: MARTINEZ Edgar

FractalDB code based original idea: https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/


## Requirements on ABCI

This repository is only tested on V100 nodes yet.
- Modules: 

    1) cuda/12.0/12.0.0   2) cudnn/8.8/8.8.1   3) nccl/2.17/2.17.1-1   4) gcc/12.2.0   5) cmake/3.26.1   6) hpcx-mt/2.12

Python version: 
    Python 3.11.1 (main, Mar  2 2023, 09:44:18) [GCC 11.2.0] on linux. 

We choosed python 3.11.1 since the improvement compared to 3.10 is about 20% in some cases as for CPU operations.
We utilize "pyenv" to install several environments -> https://github.com/pyenv/pyenv

We recomend install the packages in this order:

1) Install Python 3.11.1
    1.1 Update pip

2) Install Torch 2.0.1 from the webpage "pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117" -> https://pytorch.org/get-started/previous-versions/


3) Install DALI pipelin "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120 " -> https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html

4) Install the rest of the packages from the file "requirements.txt"

## Usage
In order to keep consistency for reproducibility, we suggest to follow the next procedures in execution.

We utilized an interactive V100 with the following command -> qrsh -g gcc50533 -l rt_F=1 -l h_rt=12:00:00 -l USE_SSH=1 -v SSH_PORT=2299

First, clone this repository or copy somewhere.
Second, locate your local SSD and create a symbolic link to your workspace -> " ln -s $SGE_LOCALDIR ssd ".

The following steps are creation of dataset and measurement.

### Creation of large FractalDB using GPUs -> Headless rendering using EGL.
Utilize the following command to create FractalDB on local SSD. You can modify as you want in order to create the dataset in different locations using "save_root" option.

mpirun --bind-to socket --use-hwthread-cpus -np 80 python mpi_createFDB_gpu.py -g 4 --image-res 362 --iteration 100000 --save_root ssd/fdb1k

This will create a FractalDB-1k inlcuding 1 million images.
### I/O performance test using PyTorch
To run I/O mesurements reading the dataset we followed two simple conditions in the experiment:

1) The files are retrivewd using the "imageFolder" class structure from TorchVision. This is the most common way to retrieve image datasets using Pytorch.

2) We measured performance taking into account that the reading process start from retrieving the image from the physical space (ssd/nfs) and until it reaches the GPU (ToTensor() operation on the GPU.) ready for training. This includes a basic transformation operations, be aware.

mpirun --bind-to socket -np 4 python loadingdataset_benchmark.py --root ssd/FractalDB-1000-EGL-GLFW -b 100 -j 19 --epochs 1 --log-interval 1000 -d 

## Author measurement

I utilize:

- "htop" to check CPU occupancy.
- "watch -n 0.1 free -h -l -w" to check RAM memory pressure.
- "nvidia-smi -lms 1000" to check GPU related information.
- "watch -n 1 df -h" to check local storage capacity.
