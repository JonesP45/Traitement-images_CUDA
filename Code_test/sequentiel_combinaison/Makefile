CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_seq_comb: main_seq_comb.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_seq_comb
