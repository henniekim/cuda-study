#include <stdio.h>
#define N 65536

__global__ void add( int *a, int *b, int *c )
{
	int tid = blockIdx.x; // CUDA 런타임의 내장(built-in)변수 중 하나
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c; // 포인터 변수 선언

	// GPU 메모리를 할당한다.
	cudaMalloc((void**)&dev_a, N * sizeof(int)); // CUDA에서 메모리를 할당하는 방법 : cudaMallocMAnaged 도 있음 
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) // CPU 연산을 이용하여 Matrix값 초기화
	{
		a[i] = -i;
		b[i] = i * i;
	}

	// 배열 'a'와 'b'를 CPU에서 GPU로 복사한다.
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// GPU에서 합 연산 수행
	add << <N, 1 >> > (dev_a, dev_b, dev_c); // <<<병렬블록의 개수, ? >>>

	// 배열 'c'를 GPU에서 CPU로 복사한다.
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// 결과를 출력한다.
	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}