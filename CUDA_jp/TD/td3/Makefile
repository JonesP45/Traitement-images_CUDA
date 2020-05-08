CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

grayscale: grayscale.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm grayscale
