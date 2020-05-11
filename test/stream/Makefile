CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_stream: main_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_stream
