CXX=nvcc
LDLIBS=`pkg-config --libs opencv`


mainshared: mainshared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main
