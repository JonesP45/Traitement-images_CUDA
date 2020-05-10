CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main_shared: main_shared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm main_shared
