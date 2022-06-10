#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

void query_device() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	printf("Device count: %d\n", deviceCount);

	int devNo = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);
	
	printf("Device %d: %s\n", devNo, iProp.name);
	printf("\tNumber of multiprocessors:\t\t%d\n", iProp.multiProcessorCount);
	printf("\tClock rate:\t\t\t\t%d\n", iProp.clockRate);
	printf("\tCompute capability:\t\t\t%d.%d\n", iProp.major, iProp.minor);
	printf("\tGlobal Memmory:\t\t\t\t%4.2f KB\n", iProp.totalGlobalMem / 1024.0);
	printf("\tConstant Memmory:\t\t\t%4.2f KB\n", iProp.totalConstMem / 1024.0);
	printf("\tShared Memmory per block:\t\t%4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
	printf("\tShared Memmory per MP:\t\t\t%4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
	printf("\tRegiters per block:\t\t\t%d\n", iProp.regsPerBlock);
	printf("\tWarp Size:\t\t\t\t%d\n", iProp.warpSize);
	printf("\tMaximum number of threads per block:\t%d\n", iProp.maxThreadsPerBlock);
	printf("\tMaximum number of threads per MP:\t%d\n", iProp.maxThreadsPerMultiProcessor);
	printf("\tMaximum number of warps per MP:\t\t%d\n", iProp.maxThreadsPerMultiProcessor / 32);
	printf("\tMaximum Grid Size:\t\t\t(%d,%d,%d)\n", iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
	printf("\tMaximum Block Dimesion:\t\t\t(%d,%d,%d)\n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
}

/* Results:
Device count: 1
Device 0: NVIDIA GeForce RTX 3070 Laptop GPU
		Number of multiprocessors:              40
		Clock rate:                             1290000
		Compute capability:                     8.6
		Global Memmory:                         8388096.00 KB
		Constant Memmory:                       64.00 KB
		Shared Memmory per block:               48.00 KB
		Shared Memmory per MP:                  100.00 KB
		Regiters per block:                     65536
		Warp Size:                              32
		Maximum number of threads per block:    1024
		Maximum number of threads per MP:       1536
		Maximum number of warps per MP:         48
		Maximum Grid Size:                      (2147483647,65535,65535)
		Maximum Block Dimesion:                 (1024,1024,64)
*/

int main() {
	query_device();
	return 0;
}