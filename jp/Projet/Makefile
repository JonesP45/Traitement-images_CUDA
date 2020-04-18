CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

grayscale: blur_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm grayscale
