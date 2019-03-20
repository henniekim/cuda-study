// CUDA 공부를 위한 예제 코드 분석 
// 책 내용과 구글 곳곳에 검색해서 알아낸 내용 정리

#include "./common/book.h"

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 ;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin (32, (N+threadsPerBlock-1) / threadsPerBlock); // TODO:.  이 코드의 의미는 무엇일까?

// N개의 데이터 원소들이 있을 때 내적을 하려면 오직 N개의 스레드가 필요하다.
/*

    |0|1|2|3|4|5|6|...|N-1|N|
A : | | | | | | | |   |   | | 
             DOT
B : | | | | | | | |   |   | |
              =
C : | | | | | | | |   |   | |

    ---- needs N-threads ----

*/
// N 이상의 가장 작은 threadsPerBlock 배수가 필요하다. (더 많아도 되지만 효율성 측면에서 ! )
// 256 개 스레드 * 32개 블록 

__global__ void dot( float *a, float *b, float *c) // 왜 이름이 void kernel이 아닐까?
{
    __shared__ float cache[threadsPerBlock]; // 블락 당 스레드 개수 : 상단에서 256개로 정의함
    // 공유메모리는 블록 내의 스레드들이 계산을 하는 데 통신과 협력을 할 수 있게 한다.
    // 공유 메모리 버퍼는 물리적으로 GPU상에 상주한다. : 공유 메모리의 접근 지연 시간은 매우 짧다.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  
    // 블록 0 | 스레드 0 스레드 1 스레드 2 스레드 3 |
    // 블록 1 | 스레드 4 스레드 5 스레드 6 스레드 7 |
    // 블록 2 | 스레드 8 스레드 9 스레드10 스레드11 | 
    // 블록 3 | 스레드12 스레드13 스레드14 스레드15 |
    //         ---------- blockDim.x ---------
    int cacheIndex = threadIdx.x; // 캐쉬 인덱스는 스레드 인덱스와 같다

    float temp = 0;
    while (tid < N )
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x; // 몇 칸씩 인덱스를 건너 뛸 것인지 
    }
    // 캐시 값들을 설정한다.
    cache[cacheIndex] = temp;

    // 블록의 스레드 들을 동기화한다.
    __syncthreads();

    // 다음의 코드 때문에 threadsPerBlock 이 2의 제곱수이어야 한다... TODO: 는데 잘 모르겠다 (WHY?)
    int i = blockDim.x/2;
    while ( i != 0 )
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
            __syncthreads();
            i /= 2;
    }

    if (cacheIndex == 0)
    c[blockIdx.x] = cache[0]; // TODO: 왜 cache 인덱스 0만 접근하는지
}

int main ( void ) 
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // CPU memory 할당
    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float)); // N = 33 * 1024
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

    // GPU memory 할당
    HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));

    // 호스트에 단순이 데이터를 채우는 작업, 또는 이미지 데이터를 행렬로 불러오는 것으로 대체할 수도 있을 듯
    for (int i=0; i<N; i++)
    {
        a[i] = i;
        b[i] = i * 2; 
    }

    // 배열을 CPU에서 GPU로 옮겨준다. (실제 프로그램을 짤때 여기서 발생하는 병목을 주의해야 한다 !)
    // GPU-> GPU로 복사하는것은 상대적으로 빠르지만, 디바이스트와 호스트를 서로 오가는 메모리는 상대적으로 느림
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    // CPU에서 마무리 작업
    c = 0; // 초기화 하고
    for (int i=0; i<blocksPerGrid; i++)
    {
        // printf("%f\n", c);
        // printf("%f\n", partial_c[i]); // just for debugging 
        c += partial_c[i]; // 합을 CPU를 이용해서 구한다. 병렬처리 하기에는 자원의 낭비가 크기 때문.
    }

    // TODO: 이건 또 무슨 의미일까...  SOLVED -> 그냥 합구하는 공식, GPU 합과 비교하기 위한 검증 장치
    #define sum_squares(x) (x*(x+1)*(2*x+1) /6)
    printf( "Does GPU value %.6g = %.6g ?\n", c, 2 * sum_squares((float)(N-1))); // 

    cudaFree( dev_a);
    cudaFree( dev_b);
    cudaFree( dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
}