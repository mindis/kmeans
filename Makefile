CXXFLAGS=-march=native -O3 -g -std=c++11 -Wall -Wpedantic -lpthread #-funsafe-math-optimizations # -ffast-math #-fopt-info-vec-missed 

cluster: cluster.cpp

clean:
	rm cluster