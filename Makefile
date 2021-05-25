include Makefile.inc

all: main

main : src/main.cu src/sssp_cpu.hxx src/sssp_1gpu.hxx src/sssp_mgpu.hxx
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} -Inccl/build/include -Lnccl/build/lib -lnccl --compiler-options "${CXXFLAGS}" -o main src/main.cu

clean:
	rm -f main