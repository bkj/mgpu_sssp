#!/bin/bash

# install.sh

# --
# Setup python environment (for preprocessing)

conda create -y -n sssp_env python=3.7
conda activate sssp_env

pip install numpy==1.20.3
pip install scipy==1.6.3

# --
# Prep data

python prob2bin.py --inpath data/chesapeake.mtx

# --
# Install NCCL
# !! Maybe you can skip this, if NCCL is already installed on your system

wget https://github.com/NVIDIA/nccl/archive/refs/tags/v2.9.6-1.tar.gz
tar -xzvf v2.9.6-1.tar.gz
rm v2.9.6-1.tar.gz
mv nccl-2.9.6-1 nccl
cd nccl
make -j12

# --
# Compile

make clean
make main

# run w/ one seed
./main data/chesapeake.bin

# run w/ 10 seeds
./main data/chesapeake.bin 10

# --
# Run on larger datasets

function fetch_rmat {
  SCALE=$1
  wget https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale${SCALE}-ef16/graph500-scale${SCALE}-ef16_adj.mmio.gz
  gunzip graph500-scale${SCALE}-ef16_adj.mmio.gz
  mv graph500-scale${SCALE}-ef16_adj.mmio data/rmat${SCALE}.mtx
}

SCALE=18
fetch_rmat $SCALE
python prob2bin.py --inpath data/rmat18.mtx
./main data/rmat18.bin