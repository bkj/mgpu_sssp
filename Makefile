include Makefile.inc

all: sssp cusssp

sssp: sssp.cpp
	g++ $(CXXFLAGS) -o sssp sssp.cpp

cusssp : cusssp.cu
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o cusssp cusssp.cu $(SOURCE) $(ARCH) $(INC)

clean:
	rm -f sssp cusssp