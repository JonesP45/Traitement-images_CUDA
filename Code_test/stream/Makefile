CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

stream: stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

stream_test: stream_test.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm stream

clean_test:
	rm stream_test
