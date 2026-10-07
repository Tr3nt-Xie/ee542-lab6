# CPU binaries build anywhere (OpenMP variants need gcc, not Apple clang).
# GPU targets need nvcc (Colab, a GPU VM). ARCH=native targets the GPU in this
# machine; override, e.g. ARCH=sm_75 for a T4, if your nvcc lacks -arch=native.
ARCH ?= native
NVCC ?= nvcc
NVFLAGS = -O3 -arch=$(ARCH)

all: cpu gpu lib
cpu: bin/matmul_cpu bin/matmul_cpu_omp bin/conv_cpu bin/conv_cpu_omp
gpu: bin/sgemm bin/conv_gpu
lib: bin/liblab6.so

bin/matmul_cpu: cpu/matmul_cpu.c | bin
	$(CC) -O3 -march=native -o $@ $<
bin/matmul_cpu_omp: cpu/matmul_cpu.c | bin
	$(CC) -O3 -march=native -fopenmp -o $@ $<
bin/conv_cpu: conv/conv_cpu.c | bin
	$(CC) -O3 -march=native -o $@ $<
bin/conv_cpu_omp: conv/conv_cpu.c | bin
	$(CC) -O3 -march=native -fopenmp -o $@ $<
bin/sgemm: gpu/sgemm.cu gpu/kernels.cuh | bin
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas
bin/conv_gpu: gpu/conv_gpu.cu gpu/conv_kernels.cuh conv/conv_cpu.c | bin
	$(NVCC) $(NVFLAGS) -o $@ $<
bin/liblab6.so: python/lab6lib.cu gpu/kernels.cuh gpu/conv_kernels.cuh | bin
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $< -lcublas
bin:
	mkdir -p bin
clean:
	rm -rf bin
.PHONY: all cpu gpu lib clean
