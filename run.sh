#!/bin/bash

# run.sh

make clean
make cusssp
CUDA_VISIBLE_DEVICES=0 ./cusssp data/rmat24.bin