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

void sample2() {
	int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33 };

	for (int i = 0; i < array_size; i++) {
		printf("%d,", h_data[i]);
	}
	printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(8);
	dim3 grid(1);

	unique_idx_calc_threadIdx << <grid, block >> > (d_data);
}

int main() {
	//sample1();
	sample2();

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}