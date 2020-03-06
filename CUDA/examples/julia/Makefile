CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

all: julia julia-cu

julia: julia.cpp


julia-cu: julia.cu
	nvcc -o $@ $< $(LDLIBS)
