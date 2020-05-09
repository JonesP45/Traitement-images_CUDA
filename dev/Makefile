CXX=nvcc
LDLIBS=`pkg-config --libs opencv`


main3c: main3c.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main
