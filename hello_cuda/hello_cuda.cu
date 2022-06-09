#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void hello_cuda() {
	printf("Hello Cuda\n");
}

int main() {
	dim3 grid(8); // number of blocks
	dim3 block(4); // threads per block

	hello_cuda << <grid, block>> > ();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}