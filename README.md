# Benchmarks for I/O Performance on Synthetic Datasets (FractalDB) Using ABCI Super AI System

**Author:** Edgar MARTINEZ

FractalDB code based on the original idea: [Pretraining without Natural Images](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)

## Requirements on ABCI

This repository has been tested only on V100 nodes so far.
- **Modules:**
    1) cuda/12.0/12.0.0
    2) cudnn/8.8/8.8.1
    3) nccl/2.17/2.17.1-1
    4) gcc/12.2.0
    5) cmake/3.26.1
    6) hpcx-mt/2.12

**Python version:**
    Python 3.11.1 (main, Mar 2 2023, 09:44:18) [GCC 11.2.0] on Linux.

We chose Python 3.11.1 due to a 20% improvement in certain CPU operations compared to version 3.10. We use "pyenv" to manage multiple environments, which can be found at [pyenv GitHub](https://github.com/pyenv/pyenv).

We recommend installing the packages in the following order:

1) Install Python 3.11.1
    1.1 Update pip

2) Install Torch 2.0.1 from the webpage: 
   ```
   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
   ```
   [PyTorch Installation Guide](https://pytorch.org/get-started/previous-versions/)

3) Install DALI pipeline: 
   ```
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
   ```
   [DALI Installation Guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)

4) Install the remaining packages from the "requirements.txt" file.

## Usage

To maintain consistency for reproducibility, please follow these procedures during execution.

We used an interactive V100 with the following command: 
```
qrsh -g gcc50533 -l rt_F=1 -l h_rt=12:00:00 -l USE_SSH=1 -v SSH_PORT=2299
```

1) Clone this repository or copy it to your desired location.
2) Locate your local SSD and create a symbolic link to your workspace:
   ```
   ln -s $SGE_LOCALDIR ssd
   ```

The following steps involve creating the dataset and performing measurements.

### Creating a Large FractalDB Using GPUs (Headless Rendering Using EGL)

Use the following command to create FractalDB on the local SSD. You can modify the "save_root" option to create the dataset in different locations.
```
mpirun --bind-to socket --use-hwthread-cpus -np 80 python mpi_createFDB_gpu.py -g 4 --image-res 362 --iteration 100000 --save_root ssd/fdb1k
```

This will create FractalDB-1k, including 1 million images.

### I/O Performance Test Using PyTorch

To run I/O measurements reading the dataset, we followed two simple conditions in the experiment:

1) The files are retrieved using the "imageFolder" class structure from TorchVision. This is the most common way to retrieve image datasets using PyTorch.
2) We measured performance taking into account that the reading process starts from retrieving the image from the physical space (ssd/nfs) and continues until it reaches the GPU (ToTensor() operation on the GPU), ready for training. This includes basic transformation operations; be aware of this.

```
mpirun --bind-to socket -np 4 python loadingdataset_benchmark.py --root ssd/FractalDB-1000-EGL-GLFW -b 100 -j 19 --epochs 1 --log-interval 1000 -d
```

## Author's Measurement Tools

I use the following tools to monitor the system:

- **htop:** To check CPU occupancy.
- **watch -n 0.1 free -h -l -w:** To monitor RAM memory pressure.
- **nvidia-smi -lms 1000:** To check GPU-related information.
- **watch -n 1 df -h:** To monitor local storage capacity.