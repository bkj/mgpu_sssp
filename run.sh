#!/bin/bash

# run.sh

PROB=rmat24.bin

# --
# Build

make clean
make main -j4

# --
# Run

CUDA_VISIBLE_DEVICES=0        ./main data/$PROB
CUDA_VISIBLE_DEVICES=0,1      ./main data/$PROB
CUDA_VISIBLE_DEVICES=0,1,2    ./main data/$PROB
CUDA_VISIBLE_DEVICES=0,1,2,3  ./main data/$PROB

