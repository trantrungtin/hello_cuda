# hello_cuda

<p align="center">
    <img src="images/pic1.png" alt="Logo" width="512" height="310">
</p>

```c++
dim3 grid(2, 2); // number of blocks
dim3 block(8, 2); // threads per block
hello_cuda << <grid, block>> > ();
```

threadIdx is dim3 type variable

<p align="center">
    <img src="images/pic2.png" alt="threadIdx" width="512" height="310">
</p>

blockIdx is dim3 type variable

<p align="center">
    <img src="images/pic3.png" alt="blockIdx" width="512" height="310">
</p>

blockDim variable consist number of threads.

<p align="center">
    <img src="images/pic4.png" alt="blockDim" width="512" height="310">
</p>

gridDim variable consist number of thread blocks in each dimension of a grid

<p align="center">
    <img src="images/pic5.png" alt="gridDim" width="512" height="310">
</p>

Copy data from host to device

```c++
int array_size = 8;
int array_byte_size = sizeof(int) * array_size;
int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33 };

int* d_data;
cudaMalloc((void**)&d_data, array_byte_size);
cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
```