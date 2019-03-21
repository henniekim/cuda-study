#include <stdio.h>
#define N 65536

__global__ void add( int *a, int *b, int *c )
{
	int tid = blockIdx.x; // CUDA ��Ÿ���� ����(built-in)���� �� �ϳ�
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c; // ������ ���� ����

	// GPU �޸𸮸� �Ҵ��Ѵ�.
	cudaMalloc((void**)&dev_a, N * sizeof(int)); // CUDA���� �޸𸮸� �Ҵ��ϴ� ��� : cudaMallocMAnaged �� ���� 
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	for (int i = 0; i < N; i++) // CPU ������ �̿��Ͽ� Matrix�� �ʱ�ȭ
	{
		a[i] = -i;
		b[i] = i * i;
	}

	// �迭 'a'�� 'b'�� CPU���� GPU�� �����Ѵ�.
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// GPU���� �� ���� ����
	add << <N, 1 >> > (dev_a, dev_b, dev_c); // <<<���ĺ���� ����, ? >>>

	// �迭 'c'�� GPU���� CPU�� �����Ѵ�.
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// ����� ����Ѵ�.
	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d \n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}