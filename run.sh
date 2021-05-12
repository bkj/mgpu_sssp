#!/bin/bash

# run.sh

make clean
make cusssp
CUDA_VISIBLE_DEVICES=0 ./cusssp rmat22.bin