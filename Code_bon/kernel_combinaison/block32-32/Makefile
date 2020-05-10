CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_comb: main_comb.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_comb
