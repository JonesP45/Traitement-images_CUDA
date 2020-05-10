CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

stream: stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm stream
