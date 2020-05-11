CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_kernel: main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_kernel
