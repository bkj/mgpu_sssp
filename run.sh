#!/bin/bash

# run.sh

PROB=rmat24.bin

# --
# Build

make clean
make cusssp -j4

# --
# Run

CUDA_VISIBLE_DEVICES=0        ./cusssp data/$PROB
CUDA_VISIBLE_DEVICES=0,1      ./cusssp data/$PROB
CUDA_VISIBLE_DEVICES=0,1,2    ./cusssp data/$PROB
CUDA_VISIBLE_DEVICES=0,1,2,3  ./cusssp data/$PROB

