#include "cuda.h"
#include "../common/cpu_bitmap.h"
#include "../common/book.h"

#define RND(n) (n*rand() / RAND_MAX)
#define DIM 1024
#define SPHERES 50
#define INF 2e10f

struct Sphere 
{
    float r,g,b; // 구의 색깔
    float radius; // 구의 반지름 길이
    float x,y,z; // 구의 중심좌표 위치 

    // ox = x - DIM / 2
    // oy = y - DIM / 2 

    __device__ float hit (float ox, float oy, float *n) // ox, oy : 광선을 발사할 픽셀
    {
        float dx = ox - x;  // 구의 중심과 ox와의 거리 
        float dy = oy - y;  // 구의 중심과 oy와의 거리 

        if (dx*dx + dy* dy< radius * radius ) // 구와 접촉을 한다면 
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius) ; // 멀리 갈수록 n이 작아진다.

            return z + dz ; // = oz // 구의 중심과 oz와의 거리 // 카메라의 위치로 부터 광선이 구와 닿는 지점까지의 거리  
        }

        return -INF;
    }
};

__constant__ Sphere s[SPHERES];


__global__ void kernel (unsigned char *ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = (x-DIM/2);
    float oy = (y-DIM/2);

    float r=0, g=0, b=0;
    float maxz = -INF;

    for (int i=0; i<SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);

        if ( t > maxz )
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;

            maxz = t ;
        }
    }
    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

  


int main(void)
{
    cudaEvent_t start , stop;
    HANDLE_ERROR( cudaEventCreate( &start));
    HANDLE_ERROR( cudaEventCreate( &stop));
    HANDLE_ERROR( cudaEventRecord( start, 0));

    CPUBitmap bitmap( DIM, DIM );
    unsigned char *dev_bitmap;


    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size()));

    HANDLE_ERROR( cudaMalloc( (void**)&s, sizeof(Sphere) * SPHERES));


    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);

    for (int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = RND( 1.0f);
        temp_s[i].g = RND( 1.0f);
        temp_s[i].b = RND( 1.0f);
        temp_s[i].x = RND( 1000.0f) - 500;
        temp_s[i].y = RND( 1000.0f) - 500;
        temp_s[i].z = RND( 1000.0f) - 500;
        temp_s[i].radius = RND( 100.0f ) + 20;
    }

    HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES));

    free(temp_s);


    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);
    kernel<<<grids, threads>>>(dev_bitmap);

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));


    HANDLE_ERROR( cudaEventRecord(stop, 0 ));
    HANDLE_ERROR( cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop));

    printf(" Time to generate : %3.1f ms\n", elapsedTime);

    HANDLE_ERROR( cudaEventDestroy( start));
    HANDLE_ERROR( cudaEventDestroy( stop));

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

    cudaFree(s);

}
