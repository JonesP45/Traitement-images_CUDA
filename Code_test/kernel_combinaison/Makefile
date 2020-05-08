CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: main_comb.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main
