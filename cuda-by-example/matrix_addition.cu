#include <stdio.h>
#define SIZE 1024

__global__ void VectorAdd(int *a, int *b, int *c, int n) // __global__ �� �� �Լ��� GPU������ ���� �� ������ �˷���
{
	int i = threadIdx.x; // read only variable

	if (i < n)
		c[i] = a[i] * b[i];
	//int i; // ���� for������ �ۼ��ϸ� �̷��� �ȴ�.
	//for (i = 0; i < n; ++i)
	//	c[i] = a[i] + b[i];
}


int main()
{
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int)); // cuda ���� �޸� �Ҵ��ϴ� ���
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; i++) // ��Ʈ������ CPU�� �̿��ؼ� �ʱ�ȭ �Ѵ�. (������ �����͸� �������� ������ ����) 
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

	cudaFree(a); // free ��ſ� cudaFree�� GPU�� �Ҵ��� �޸� ����
	cudaFree(b);
	cudaFree(c);

	return 0;
}