CXX=nvcc
LDLIBS=`pkg-config --libs opencv`


sobel-fusion2: sobel-fusion2.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main
