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

__global__ void hello_cuda() {
	printf("Hello Cuda blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d <> threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d\n", 
		blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z);
}

void sample1() {
	dim3 grid(2, 2); // number of blocks
	dim3 block(8, 2); // threads per block

	hello_cuda << <grid, block >> > ();
}

__global__ void unique_idx_calc_threadIdx(int* input) {
	int tid = threadIdx.x;
	printf("threadIdx: %d, value: %d\n", tid, input[tid]);
}

__global__ void unique_gid_calculation(int* input) {
	int tid = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int gid = tid + offset;
	printf("blockIdx.x=%d, threadIdx.x=%d, gid=%d, value=%d\n", blockIdx.x, threadIdx.x, gid, input[gid]);
}

void createCudaData(int* h_data, size_t size, int** d_data) {
	if (h_data == nullptr) {
		return;
	}
	gpuErrchk(cudaMalloc((void**)d_data, size));
	gpuErrchk(cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice));
}

void print(int* arr, size_t size) {
	size_t cnt = size / sizeof(int);
	for (int i = 0; i < cnt; i++) {
		printf("%d,", arr[i]);
	}
	printf("\n\n");
}

void sample2() {
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33 };
	print(h_data, sizeof(h_data));

	int* d_data;
	createCudaData(h_data, sizeof(h_data), &d_data);

	dim3 block(4);
	dim3 grid(2);

	//unique_idx_calc_threadIdx << <grid, block >> > (d_data);
	unique_gid_calculation << <grid, block >> > (d_data);
}

__global__ void unique_gid_calculation_2d(int* input) {
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x;
	int row_offset = gridDim.x * blockDim.x * blockIdx.y;
	int gid = row_offset + block_offset + tid;
	printf("blockIdx.x=%d, threadIdx.x=%d, gid=%d, value=%d\n", 
		blockIdx.x, threadIdx.x, gid, input[gid]);
}

void sample3() {
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33, 22, 43, 56, 4, 76, 81, 94, 32 };
	print(h_data, sizeof(h_data));
	int* d_data;
	createCudaData(h_data, sizeof(h_data), &d_data);
	dim3 block(4);
	dim3 grid(2, 2);
	unique_gid_calculation_2d << <grid, block >> > (d_data);
}

__global__ void unique_gid_calculation_2d_2d(int* input) {
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int number_threads_in_a_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * number_threads_in_a_block;

	int number_threads_in_a_row = number_threads_in_a_block * gridDim.x;
	int row_offset = number_threads_in_a_row * blockIdx.y;
	int gid = row_offset + block_offset + tid;
	printf("blockIdx.x=%d, threadIdx.x=%d, gid=%d, value=%d\n",
		blockIdx.x, threadIdx.x, gid, input[gid]);
}

void sample4() {
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33, 22, 43, 56, 4, 76, 81, 94, 32 };
	print(h_data, sizeof(h_data));
	int* d_data;
	createCudaData(h_data, sizeof(h_data), &d_data);
	dim3 block(2, 2);
	dim3 grid(2, 2);
	unique_gid_calculation_2d_2d << <grid, block >> > (d_data);
}

__global__ void unique_gid_calculation_3d(int* input) {
	int gid = blockDim.x * blockDim.y * threadIdx.z
		+ blockDim.x * threadIdx.y
		+ threadIdx.x
		;
	printf("tid=%d, gid=%d, value=%d\n", threadIdx.x, gid, input[gid]);
}

void createRandData(int** data, size_t number_of_elements) {
	size_t byte_size = sizeof(int) * number_of_elements;
	(*data) = (int*)malloc(byte_size);
	time_t t;
	srand((unsigned)time(&t));
	for (size_t i = 0; i < number_of_elements; i++) {
		(*data)[i] = (int)(rand() & 0xff);
	}
}

void sample5() {
	int size = 8;
	int byte_size = size * sizeof(int);
	int* h_data;
	createRandData(&h_data, size);
	print(h_data, byte_size);
	int* d_data;
	createCudaData(h_data, byte_size, &d_data);
	dim3 block(2, 2, 2);
	dim3 grid(1,1,1);
	unique_gid_calculation_3d << <grid, block >> > (d_data);
}

__global__ void unique_gid_calculation_3d_3d(int* input) {
	int number_threads_in_a_block = blockDim.x * blockDim.y * blockDim.z;
	int number_threads_in_a_plan = number_threads_in_a_block * gridDim.x * gridDim.y;
	int gid = number_threads_in_a_plan * blockIdx.z
		+ number_threads_in_a_block * gridDim.x * blockIdx.y
		+ number_threads_in_a_block * blockIdx.x
		+ blockDim.x * blockDim.y * threadIdx.z
		+ blockDim.x * threadIdx.y
		+ threadIdx.x
		;
	printf("block(%d,%d,%d) thread(%d,%d,%d) gid=%d, value=%d\n",
		blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z,
		gid, input[gid]);
}

void sample6() {
	int size = 64;
	int byte_size = size * sizeof(int);
	int* h_data;
	createRandData(&h_data, size);
	print(h_data, byte_size);
	int* d_data;
	createCudaData(h_data, byte_size, &d_data);
	dim3 block(2, 2, 2);
	dim3 grid(2, 2, 2);
	unique_gid_calculation_3d_3d << <grid, block >> > (d_data);
}

__global__ void sum_array_gpu(int* a, int* b, int* c, int number_of_elements) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < number_of_elements) {
		c[gid] = a[gid] + b[gid];
	}
}

void sum_array_cpu(int* a, int* b, int* c, int number_of_elements) {
	for (int i = 0; i < number_of_elements; i++) {
		c[i] = a[i] + b[i];
	}
}

void compare_arrays(int* a, int* b, int number_of_elements) {
	for (int i = 0; i < number_of_elements; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different.\n");
			return;
		}
	}
	printf("Arrays are same.\n");
}

void sum_two_arrays() {
	const size_t number_of_elements = 5000000;
	const size_t number_of_bytes = number_of_elements * sizeof(int);
	const size_t block_size = 128;

	int* h_a, * h_b, * h_c, * gpu_results;
	createRandData(&h_a, number_of_elements);
	createRandData(&h_b, number_of_elements);
	h_c = (int*)malloc(number_of_bytes);
	gpu_results = (int*)malloc(number_of_bytes);

	clock_t htod_start, htod_end;
	htod_start = clock();
	int* d_a, * d_b, * d_c;
	createCudaData(h_a, number_of_bytes, &d_a);
	createCudaData(h_b, number_of_bytes, &d_b);
	gpuErrchk(cudaMalloc((void**)&d_c, number_of_bytes));
	htod_end = clock();

	// launching the grid
	dim3 block(block_size);
	dim3 grid(number_of_elements / block_size + 1);

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, number_of_elements);
	gpuErrchk(cudaDeviceSynchronize());
	gpu_end = clock();

	// memory transfer back to host
	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	gpuErrchk(cudaMemcpy(gpu_results, d_c, number_of_bytes, cudaMemcpyDeviceToHost));
	dtoh_end = clock();

	// array comparison
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_array_cpu(h_a, h_b, h_c, number_of_elements);
	cpu_end = clock();
	compare_arrays(h_c, gpu_results, number_of_elements);

	printf("Sum array CPU execution time: %4.6f\n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
	printf("Sum array GPU execution time: %4.6f\n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
	printf("Transfer data from host to device: %4.6f\n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
	printf("Transfer data from device to host: %4.6f\n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));

	// cleanup data
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);
	free(gpu_results);
}

int main() {
	//sample1();
	//sample2();
	//sample3();
	//sample4();
	//sample5();
	//sample6();
	sum_two_arrays();

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}