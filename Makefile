CXXFLAGS += -std=c++11 -mtune=native -march=native -Wall -Wunused-result -O3 -DNDEBUG -g -fopenmp -ffast-math

APP=sssp

all: $(APP)

$(APP): $(APP).cpp
	g++ $(CXXFLAGS) -o $(APP) $(APP).cpp

clean:
	rm -f $(APP)