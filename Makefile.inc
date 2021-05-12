NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

CXX = g++

GEN_SM70    = -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100
SM_TARGETS = $(GEN_SM70)

NVCCFLAGS   += -std=c++14 -lnvToolsExt
NVCCFLAGS   += $(SM_TARGETS)
NVCCFLAGS   += --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math --ptxas-options -v --relocatable-device-code true
NVCCFLAGS   += -O3 --generate-line-info

CXXFLAGS    += -std=c++14 
CXXFLAGS    += -fopenmp
CXXFLAGS    += -Wall -Wno-unused-local-typedefs -Wno-strict-aliasing -Wno-unused-function -Wno-format-security
CXXFLAGS    += -O3