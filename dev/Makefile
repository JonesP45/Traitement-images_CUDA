CXX=nvcc
LDLIBS=`pkg-config --libs opencv`


mainshared3c: mainshared3c.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm main
