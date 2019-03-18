#include "./common/book.h"
#include "./common/gl_helper.h"
#include "./common/cpu_bitmap.h"
#include "./common/cpu_anim.h"

#define DIM 1024  // 이미지 사이즈 정의

__global__ void kernel( unsigned char *ptr, int ticks)
{
    // threadIdx / blockIdx 로 픽셀 위치를 결정한다.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x; 

    float fx = x - DIM/2; // center 를 맞추기 위함
    float fy = y - DIM/2;

    float d = sqrt( fx*fx + fy*fy);
    unsigned char grey = (unsigned char)(128.0f+127.0f*cos(d/10.0f-ticks/7.0f) / (d/10.0f + 1.0f)); 
    // 이미지 픽셀의 값이기 때문에 항상 Uchar (8bit) 형태여야 한다.

    ptr[offset*4+0]=grey;
    ptr[offset*4+1]=grey;
    ptr[offset*4+2]=grey;
    ptr[offset*4+3]=255; // Alpha 값은 255로  ! 
}

struct DataBlock
{
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};


void cleanup(DataBlock*d)
{
    cudaFree( d->dev_bitmap);
}

void generate_frame( DataBlock *d, int ticks)
{
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16,16);

    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}


int main(void)
{
    DataBlock data;
    CPUAnimBitmap bitmap (DIM, DIM, &data); // 이미지 사이즈 정의
    data.bitmap = &bitmap;
    HANDLE_ERROR( cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size())); // 이미지 사이즈 만큼 GPU 메모리 할당

    bitmap.anim_and_exit(( void(*)(void*,int))generate_frame, (void(*)(void*))cleanup); // 매 프레임 생성하는 작업은 gl_helper가 해준다. 

    return 0;
}