#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void hello_cuda() {
	printf("Hello Cuda blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d <> threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d\n", 
		blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
	dim3 grid(2, 2); // number of blocks
	dim3 block(8, 2); // threads per block

	hello_cuda << <grid, block>> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}