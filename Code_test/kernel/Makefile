CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main
