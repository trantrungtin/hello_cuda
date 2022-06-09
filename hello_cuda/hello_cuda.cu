#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void hello_cuda() {
	printf("Hello Cuda x=%d, y=%d, z=%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
	dim3 grid(2, 2); // number of blocks
	dim3 block(8, 2); // threads per block

	hello_cuda << <grid, block>> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}