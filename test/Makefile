CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

all: main_seq main_kernel main_shared main_stream

main_seq: sequentiel/main_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_seq $(FILE) $(EXTENTION)

main_kernel: kernel/main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_kernel $(FILE) $(EXTENTION)

main_shared: shared/main_shared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_shared $(FILE) $(EXTENTION)

main_stream: stream/main_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_stream $(FILE) $(EXTENTION)

clean:
	rm main_seq main_kernel main_shared main_stream
