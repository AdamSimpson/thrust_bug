all: acc stream

acc: acc.cc
	nvcc -c thrust.cu
	pgc++ -acc -L/usr/local/cuda/lib64 -lcudart acc.cc thrust.o -o acc.out
stream: stream.cc
	nvcc stream.cc thrust.cu --default-stream per-thread -o stream.out
