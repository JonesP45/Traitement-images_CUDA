CXX=nvcc
LDLIBS=`pkg-config --libs opencv`

FILE=in
EXTENTION=jpg

SEQ=sequentiel/
KERNEL=kernel/
SHARED=shared/
STREAM=stream/
SHARED_STREAM=shared_stream/

all: main_seq main_kernel main_shared main_stream main_shared_stream

main_seq:
	cd $(SEQ) && $(MAKE) clean && $(MAKE)

main_kernel:
	cd $(KERNEL) && $(MAKE) clean && $(MAKE)

main_shared:
	cd $(SHARED) && $(MAKE) clean && $(MAKE)

main_stream:
	cd $(STREAM) && $(MAKE) clean && $(MAKE)

main_shared_stream:
	cd $(SHARED_STREAM) && $(MAKE) clean && $(MAKE)


