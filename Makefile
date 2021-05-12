include Makefile.inc

all: sssp cusssp test

sssp: sssp.cpp
	g++ $(CXXFLAGS) -o sssp sssp.cpp

cusssp : cusssp.cu
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} -Inccl/build/include -Lnccl/build/lib -lnccl --compiler-options "${CXXFLAGS}" -o cusssp cusssp.cu

test : test.cu
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} -Inccl/build/include -Lnccl/build/lib -lnccl --compiler-options "${CXXFLAGS}" -o test test.cu

clean:
	rm -f sssp cusssp test