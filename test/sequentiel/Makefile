CXX=nvcc
LDLIBS=`pkg-config --libs opencv`
FILE=in
EXTENTION=jpg

all: main_seq

main_seq: main_seq.cu
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)
	./main_seq $(FILE) $(EXTENTION)

clean:
	rm main_seq
