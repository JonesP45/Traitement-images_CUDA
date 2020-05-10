CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

main: main_seq.o main_kernel.o main_shared.o main_stream.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

main_seq.o: main_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

main_kernel.o: main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

main_shared.o: main_shared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

main_stream.o: main_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm *.o main
