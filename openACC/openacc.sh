#!/bin/bash
# For ubuntu 20.10

wget https://developer.download.nvidia.com/hpc-sdk/21.2/nvhpc_2021_212_Linux_x86_64_cuda_11.2.tar.gz
tar xpzf nvhpc_2021_212_Linux_x86_64_cuda_11.2.tar.gz
nvhpc_2021_212_Linux_x86_64_cuda_11.2/install

echo "export PATH=$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/compilers/bin" >> ~/.bashrc
