make clean
make cusssp
CUDA_VISIBLE_DEVICES=0        ./cusssp rmat20.bin
CUDA_VISIBLE_DEVICES=0,1      ./cusssp rmat20.bin
CUDA_VISIBLE_DEVICES=0,1,2    ./cusssp rmat20.bin
CUDA_VISIBLE_DEVICES=0,1,2,3  ./cusssp rmat20.bin