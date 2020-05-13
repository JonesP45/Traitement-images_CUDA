CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

main_shared_stream: main_shared_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_stream $(FILE) $(EXTENTION)

clean:
	rm -f main_shared_stream
