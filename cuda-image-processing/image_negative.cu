// MY Fisrt CUDA using image processing
// Image Read / Write library is following
// CUDA 코드로 진행하는 영상처리 
// 이미지 입출력을 위해 다음 라이브러리를 사용하였다.
// https://github.com/nothings/stb

#define STB_IMAGE_IMPLEMENTATION // 전처리기로 하여금 헤더 파일을 관련된 정의 소스 코드만 포함하도록 하여 효과적으로 컴파일 하게 한다.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"
#include "../common/stb_image.h"

// TODO: 큰 이미지에 대해서도 해보기 !!

__global__ void kernel( unsigned char *imageData )
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // image 픽셀을 반전한다.
    imageData[offset*3 + 0] = 255 - imageData[offset*3 + 0]; 
    imageData[offset*3 + 1] = 255 - imageData[offset*3 + 1]; 
    imageData[offset*3 + 2] = 255 - imageData[offset*3 + 2]; 

    __syncthreads(); // 이 함수는 각 브록의 스레드들을 동기화하여, 픽셀을 쓰는 작업이 끝나지 않았음에도 불구하고 데이터를 접근하여 이미지가 손상되는 현상을 막는다.
}
int main (void)
{

    int width, height, nrChannels;
    // TODO: 이미지가 없을 경우 에러 반환하는 코드 추가하기

    // 왜 이미지 불러오는 함수가 절대경로만 인식하는 것인지 잘 모르겠음 
    unsigned char *data = stbi_load("/home/donghyun/Dropbox/Repo/cuda/cuda-study/img/lena_gray.bmp", &width, &height, &nrChannels, 0);

    printf("image size is : %d x %d\n", width, height);
    printf("image channel is : %d\n", nrChannels);
    printf("image has been successfully loaded ! \n");

    int imageArraySize = width*height*nrChannels; // 불러온 이미지의 사이즈 정보를 만든다.
    unsigned char *dev_image;
    cudaMalloc((void**)&dev_image, imageArraySize*sizeof(char)); // Device에 이미지 크기 만큼의 메모리를 할당한다.
    cudaMemcpy( dev_image, data, imageArraySize*sizeof(char), cudaMemcpyHostToDevice); // CPU로 불러들인 이미지 데이터를 Device 메모리로 복사한다.
    // 병목 현상이 일어나지 않도록 주의해야 한다. (CPU - GPU 메모리 왔다갔다 하는 작업이 너무 많을 경우 주의 !)
    
    dim3 grid(width, height);
    kernel<<<grid, 256>>>( dev_image ); 

    unsigned char *result = (unsigned char*)malloc(imageArraySize*sizeof(char));

    cudaMemcpy( result, dev_image, imageArraySize*sizeof(char), cudaMemcpyDeviceToHost);

    stbi_write_bmp("/home/donghyun/Dropbox/Repo/cuda/cuda-study/img/result2.bmp", width, height, nrChannels, result); 

    cudaFree(dev_image);
    free(result);
    free(data);

    return 0;
}

