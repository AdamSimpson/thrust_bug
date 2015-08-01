all: acc stream

acc: acc.cc
	nvcc -c thrust.cu
	pgc++ -acc -L/usr/local/cuda/lib64 -lcudart acc.cc thrust.o -o acc.out
stream: stream.cc
	nvcc -c thrust.cu
	pgc++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart stream.cc thrust.o -o stream.out
