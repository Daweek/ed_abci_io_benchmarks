# A couple of benchmarks to measure I/O on synthetic datasets (FractalDB) using ABCI super AI system.

Author: MARTINEZ Edgar

FractalDB code based on original idea: https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/


## Requirements on ABCI

This repository is only tested on V100 nodes yet.
- Modules: 1) cuda/12.0/12.0.0   2) cudnn/8.8/8.8.1   3) nccl/2.17/2.17.1-1   4) gcc/12.2.0   5) cmake/3.26.1   6) hpcx-mt/2.12

Python version: Python 3.11.1 (main, Mar  2 2023, 09:44:18) [GCC 11.2.0] on linux. 
We choosed python 3.11.1 since the improvement compared to 3.10 is about 20% in some cases as for CPU operations.

For packages related to pip check requirements.txt

