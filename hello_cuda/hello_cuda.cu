#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

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
	cudaMalloc((void**)d_data, size);
	cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice);
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

int main() {
	//sample1();
	//sample2();
	sample3();

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}