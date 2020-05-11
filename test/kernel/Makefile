CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

main_kernel: main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_kernel $(FILE) $(EXTENTION)

clean:
	rm main_kernel
