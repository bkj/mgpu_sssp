#!/bin/bash

# run.sh

PROB=rmat22.bin

# --
# Build

make clean
make cusssp

# --
# Run

CUDA_VISIBLE_DEVICES=0        ./cusssp $PROB
CUDA_VISIBLE_DEVICES=0,1      ./cusssp $PROB
CUDA_VISIBLE_DEVICES=0,1,2    ./cusssp $PROB
CUDA_VISIBLE_DEVICES=0,1,2,3  ./cusssp $PROB