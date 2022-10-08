CC=gcc
NVCC=nvcc
CFLAGS=-lm 
MPFLAGS=-lm -fopenmp 
CUDAFLAGS=-lm
EXECUTABLES=c_1vp c_maxvp omp_1vp omp_maxvp cuda_1vp cuda_maxvp

main: c omp cuda

c: c.o
	$(CC) c_1vp.o $(CFLAGS) -o c_1vp
	$(CC) c_maxvp.o $(CFLAGS) -o c_maxvp

c.o: 
	$(CC) -c c_1vp.c $(CFLAGS) -o c_1vp.o
	$(CC) -c c_maxvp.c $(CFLAGS) -o c_maxvp.o

omp: omp.o
	$(CC)  omp_1vp.o $(MPFLAGS) -o omp_1vp
	$(CC)  omp_maxvp.o $(MPFLAGS) -o omp_maxvp

omp.o: 
	$(CC) -c omp_1vp.c $(MPFLAGS) -o omp_1vp.o
	$(CC) -c omp_maxvp.c $(MPFLAGS) -o omp_maxvp.o	

cuda:
	$(NVCC) -g cuda_1vp.cu $(CUDAFLAGS) -o cuda_1vp
	$(NVCC) -g cuda_maxvp.cu $(CUDAFLAGS) -o cuda_maxvp

clean:
	rm -f  *.o $(EXECUTABLES)