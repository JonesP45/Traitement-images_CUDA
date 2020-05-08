CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

main_seq_comb: main_seq_comb.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main main_seq_comb
