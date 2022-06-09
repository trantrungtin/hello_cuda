# hello_cuda

<p align="center">
    <img src="images/pic1.png" alt="Logo" width="512" height="310">
</p>

```cuda
	dim3 grid(2, 2); // number of blocks
	dim3 block(8, 2); // threads per block
    hello_cuda << <grid, block>> > ();
```