#include <stdio.h>
#define SIZE 1024

__global__ void VectorAdd(int *a, int *b, int *c, int n) // __global__ 은 이 함수가 GPU에서만 실행 될 것임을 알려줌
{
	int i = threadIdx.x; // read only variable

	if (i < n)
		c[i] = a[i] * b[i];
	//int i; // 원래 for문으로 작성하면 이렇게 된다.
	//for (i = 0; i < n; ++i)
	//	c[i] = a[i] + b[i];
}


int main()
{
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int)); // cuda 에서 메모리 할당하는 방법
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; i++) // 매트릭스는 CPU를 이용해서 초기화 한다. (임의의 데이터를 가져오는 것으로 가정) 
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	VectorAdd <<<1, SIZE>>> (a, b, c, SIZE);

	cudaDeviceSynchronize();

	int count;
	cudaGetDeviceCount(&count);

	printf("The number of GPU devices is %d\n", count);
	cudaDeviceProp prop;
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf(" --- General Information for device %d ---\n", i);
		printf(" Name : %s\n", prop.name);
		printf(" Compute capability: %d.%d\n", prop.major, prop.minor);
		printf(" Clock rate : %d \n", prop.clockRate);
		
		printf(" Total global memory : %ld MB\n", prop.totalGlobalMem/(1024*1024));
		printf(" Multiprocessor count : %d\n", prop.multiProcessorCount);
	}



	printf("\n\n");
	printf("CUDA Matrix addition example\n");
	for (int i = 0; i < 10; ++i)
		printf(" c[%d] = %d\n", i, c[i]);

	cudaFree(a); // free 대신에 cudaFree로 GPU에 할당한 메모리 해제
	cudaFree(b);
	cudaFree(c);

	return 0;
}