CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

SEQ=sequentiel/
KERNEL=kernel/
SHARED=shared/
STREAM=stream/

all: main_seq main_kernel main_shared main_stream

main_seq:
	cd $(SEQ) && $(MAKE) clean && $(MAKE)

main_kernel:
	cd $(KERNEL) && $(MAKE) clean && $(MAKE)

main_shared:
	cd $(SHARED) && $(MAKE) clean && $(MAKE)

main_stream:
	cd $(STREAM) && $(MAKE) clean && $(MAKE)

#main_seq: sequentiel/main_seq.cu
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
#
#main_kernel: kernel/main_kernel.cu
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
#	./kernel/main_kernel $(FILE) $(EXTENTION)
#
#main_shared: shared/main_shared.cu
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
#	./shared/main_shared $(FILE) $(EXTENTION)
#
#main_stream: stream/main_stream.cu
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
#	./stream/main_stream $(FILE) $(EXTENTION)
