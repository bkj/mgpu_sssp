#!/bin/bash

# run.sh

PROB=rmat20.bin
N_SEEDS=1

# --
# Build

make clean
make main -j4

# --
# Run

CUDA_VISIBLE_DEVICES=0        ./main data/$PROB $N_SEEDS
CUDA_VISIBLE_DEVICES=0,1      ./main data/$PROB $N_SEEDS
CUDA_VISIBLE_DEVICES=0,1,2    ./main data/$PROB $N_SEEDS
CUDA_VISIBLE_DEVICES=0,1,2,3  ./main data/$PROB $N_SEEDS
