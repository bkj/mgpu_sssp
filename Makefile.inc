NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

CXX = g++

GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100
SM_TARGETS = $(GEN_SM70)

CUB         = -lcub
CUSPARSE    = -lcusparse

GUNROCK_DEF = -Xcompiler -DGUNROCKVERSION=2.0.0
INC         = $(CUDA_CPP) $(CUDA_INC)

NVCCFLAGS   += -std=c++14 -lnvToolsExt
NVCCFLAGS   += $(SM_TARGETS)
NVCCFLAGS   += --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math --ptxas-options -v --relocatable-device-code true

CXXFLAGS    += -std=c++14 -fopenmp
CXXFLAGS    += -Wall
CXXFLAGS    += -Wno-unused-local-typedefs -Wno-strict-aliasing -Wno-unused-function -Wno-format-security
# CXXFLAGS += -std=c++11 -mtune=native -march=native -Wall -Wunused-result -O3 -DNDEBUG -g -fopenmp -ffast-math

# Optimization Flags
NVCCOPT     = -O3 --generate-line-info
CXXOPT      = -O3

# Debug Flags
NVCCDEBUG   = --debug --device-debug
CXXDEBUG    = -g

DEPS = ./Makefile $(SOURCE)