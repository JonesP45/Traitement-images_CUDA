CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main
