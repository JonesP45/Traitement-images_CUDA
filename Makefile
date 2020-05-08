CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: sobel-fusion2.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main
