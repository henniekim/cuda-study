#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include <stdio.h>

#define DIM 1024
#define PI 3.141592
// https://devtalk.nvidia.com/default/topic/836926/how-to-compile-codes-on-cuda-opengl-interop-from-the-book-cuda-by-example-by-jason-sanders-amp-edw/

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ float mandelbrot( int x, int y ) {
    //const float scale = 1;
    float jx = (float)(DIM/3 - x)/(DIM/3); // ensure that the real part of complex number is in the (-1,2)
    float jy = 1.2*(float)(DIM/2 - y)/(DIM/2); // ensure that the imagine part of complex number is in the (-1.2, 1.2)


    //printf("pixel coordinate is (%d, %d)", x, y);
    cuComplex a(0, 0); // initial seed 
    cuComplex c(jx, jy); // transformed coordinate (pixelcoordinate -> complex coordinate)

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        
        if (a.magnitude2() > 4)
            return 0;
    }

    //printf("%f\n", a.r);

    return 1;
}

__global__ void kernel( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    float mandelbrotValue = mandelbrot( x, y );
    //printf("mandelbrotValue is : %f \n", mandelbrotValue);
    ptr[offset*4 + 0] = 255 * (mandelbrotValue);
    ptr[offset*4 + 1] = 255 * (mandelbrotValue);
    ptr[offset*4 + 2] = 255 * (mandelbrotValue);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3 grid(DIM,DIM);
    kernel<<<grid,1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
                              
    cudaFree(dev_bitmap);
                              
    bitmap.display_and_exit();
}