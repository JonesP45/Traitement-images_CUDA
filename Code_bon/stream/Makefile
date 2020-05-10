CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

stream: main_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm stream
