#include "./common/book.h"

#define N (33*1024)

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N)
    {
        c[tid] = a[tid] +b[tid];
        tid += blockDim.x * gridDim.x; // 스레드 수 만큼 인덱스를 증가시킨다. 스레드 수 만큼 이미 동시에 실행하였으므로 건너 뛸 필요가 있다.
    }
}

int main( void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // GPU 메모리 할당!
    HANDLE_ERROR( cudaMalloc(( void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR( cudaMalloc(( void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR( cudaMalloc(( void**)&dev_c, N*sizeof(int)));

    // CPU로 배열 a와 b를 채운다. -> 이 부분을 이미지를 불러오는 과정으로 치환하면 이미지 프로세싱을 병렬 처리할 수 있지 않을까?
    for (int i = 0; i <N; i ++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

    add<<<128,128>>>(dev_a, dev_b, dev_c);

    // 배열 c를 GPU에서 CPU로 복사한다.
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

    // 성공 여부 확인
    bool success = true;
    for (int i=0; i<N; i++)
    {
        if ((a[i]+b[i]) != c[i])
        {
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }

    if (success) printf("We did it! \n");

    // GPU에 할당한 메모리 해제한다.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}