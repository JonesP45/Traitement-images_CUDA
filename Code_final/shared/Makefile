CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

main_shared: main_shared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_shared $(FILE) $(EXTENTION)

clean:
	rm -f main_shared
