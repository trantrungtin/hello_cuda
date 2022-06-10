#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void sum_arrays_gpu(int* a, int* b, int* c, int* d, size_t number_of_elements) {
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < number_of_elements) {
		d[gid] = a[gid] + b[gid] + c[gid];
	}
}

void sum_arrays_cpu(int* a, int* b, int* c, int* d, size_t number_of_elements) {
	for (size_t i = 0; i < number_of_elements; ++i) {
		d[i] = a[i] + b[i] + c[i];
	}
}

void compare_arrays(int* a, int* b, int number_of_elements) {
	for (size_t i = 0; i < number_of_elements; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different.\n");
			return;
		}
	}
	printf("Arrays are same.\n");
}

void perform(int* h_a, int* h_b, int* h_c, int* gpu_result, int* cpu_result, size_t number_of_elements, int block_size) {
	size_t number_of_bytes = number_of_elements * sizeof(int);
	int* d_a, * d_b, * d_c, * d_d;
	gpuErrchk(cudaMalloc((int**)&d_a, number_of_bytes));
	gpuErrchk(cudaMalloc((int**)&d_b, number_of_bytes));
	gpuErrchk(cudaMalloc((int**)&d_c, number_of_bytes));
	gpuErrchk(cudaMalloc((int**)&d_d, number_of_bytes));

	clock_t mem_htod_start, mem_htod_end;
	mem_htod_start = clock();
	gpuErrchk(cudaMemcpy(d_a, h_a, number_of_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, number_of_bytes, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_c, h_c, number_of_bytes, cudaMemcpyHostToDevice));
	mem_htod_end = clock();

	//kernel launch parameters
	dim3 block(block_size);
	dim3 grid((number_of_elements / block.x) + 1);

	//execution time measuring in GPU
	clock_t gpu_start, gpu_end;
	gpu_start = clock();

	sum_arrays_gpu << <grid, block >> > (d_a, d_b, d_c, d_d, number_of_elements);
	gpuErrchk(cudaDeviceSynchronize());
	gpu_end = clock();

	clock_t mem_dtoh_start, mem_dtoh_end;
	mem_dtoh_start = clock();
	gpuErrchk(cudaMemcpy(gpu_result, d_d, number_of_bytes, cudaMemcpyDeviceToHost));
	mem_dtoh_end = clock();

	compare_arrays(cpu_result, gpu_result, number_of_elements);

	printf("GPU kernel execution time sum time : %4.6f \n",
		(double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

	printf("Mem transfer host to device : %4.6f \n",
		(double)((double)(mem_htod_end - mem_htod_start) / CLOCKS_PER_SEC));

	printf("Mem transfer device to host : %4.6f \n",
		(double)((double)(mem_dtoh_end - mem_dtoh_start) / CLOCKS_PER_SEC));

	printf("Total GPU time : %4.6f \n",
		(double)((double)((mem_htod_end - mem_htod_start)
			+ (gpu_end - gpu_start)
			+ (mem_dtoh_end - mem_dtoh_start)) / CLOCKS_PER_SEC));

	// cleanup data
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_b));
	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaFree(d_d));

	cudaDeviceReset();
}

int main_sum_three_array() {
	size_t number_of_elements = 1 << 22;
	const size_t number_of_bytes = number_of_elements * sizeof(int);

	int* h_a, * h_b, * h_c, * gpu_result, * cpu_result;
	errno = 0;
	h_a = (int*)malloc(number_of_bytes);
	h_b = (int*)malloc(number_of_bytes);
	h_c = (int*)malloc(number_of_bytes);
	gpu_result = (int*)malloc(number_of_bytes);
	cpu_result = (int*)malloc(number_of_bytes);
	if (h_a == NULL || h_b == NULL || h_c == NULL || gpu_result == NULL || cpu_result == NULL) {
		printf("Could not allocation memory %d : %s\n", errno, strerror(errno));
		goto fail;
	}

	time_t t;
	srand((unsigned)time(&t));

	for (size_t i = 0; i < number_of_elements; i++) {
		h_a[i] = (int)(rand() & 0xFF);
		h_b[i] = (int)(rand() & 0xFF);
		h_c[i] = (int)(rand() & 0xFF);
	}

	memset(gpu_result, 0, number_of_bytes);
	memset(cpu_result, 0, number_of_bytes);

	// summation in CPU
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_arrays_cpu(h_a, h_b, h_c, cpu_result, number_of_elements);
	cpu_end = clock();

	printf("CPU sum time : %4.6f \n",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

	int block_sizes[] = { 64, 128, 256, 512 };
	for (size_t i = 0; i < sizeof(block_sizes) / sizeof(int); ++i) {
		int block_size = block_sizes[i];
		printf("\n\n***Block Size: %d***\n", block_size);
		perform(h_a, h_b, h_c, gpu_result, cpu_result, number_of_elements, block_size);
		memset(gpu_result, 0, number_of_bytes);
		memset(cpu_result, 0, number_of_bytes);
	}

fail:
	free(h_a);
	free(h_b);
	free(h_c);
	free(gpu_result);
	free(cpu_result);

	cudaDeviceReset();
	return 0;
}