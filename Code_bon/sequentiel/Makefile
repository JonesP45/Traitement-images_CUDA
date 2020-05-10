CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_seq: main_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_seq
