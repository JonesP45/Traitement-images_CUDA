CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

MAKE=make

FILE=in
EXTENTION=jpg

SEQ=sequentiel/

#./sequentiel/main_seq $(FILE) $(EXTENTION)
#@(cd $(SEQ) && $(MAKE))
#all: main_seq main_kernel main_shared main_stream
all: main_seq

#main_seq: sequentiel/main_seq.cu
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
main_seq:
	cd $(SEQ)
	$(MAKE)

main_kernel: kernel/main_kernel.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./kernel/main_kernel $(FILE) $(EXTENTION)

main_shared: shared/main_shared.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./shared/main_shared $(FILE) $(EXTENTION)

main_stream: stream/main_stream.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./stream/main_stream $(FILE) $(EXTENTION)

clean:
	rm main_seq main_kernel main_shared main_stream
