CXX=nvcc
LDLIBS=`pkg-config --libs opencv`


main: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main
