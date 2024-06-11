#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb
#include <iostream>
#include <ctime>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// __global__ 修饰的函数在 GPU 上执行，但是需要在 CPU 端调用
__global__
void render(unsigned char *data, int image_width, int image_height) {
    // CUDA 参数
    // blockId: 块索引, blockDim: 块内的线程数量, threadId: 线程索引, gridDim: 网格内的块数量.
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int j = index_y; j < image_height; j += stride_y) {
        for (int i = index_x; i < image_width; i += stride_x) {
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            int pixel_index = j * image_width * 3 + 3 * i;
            data[pixel_index] = ir;
            data[pixel_index + 1] = ig;
            data[pixel_index + 2] = ib;
        }
    }
}


int main() {
    // Image
    int image_width = 256;
    int image_height = 256;
    int tx = 8; // 线程数量，对应 image_width
    int ty = 8; // 线程数量，对应 image_height

    // Render
    int channels = 3; // 3通道 rgb
    unsigned char *data;
    size_t data_size = channels * image_width * image_height * sizeof(unsigned char);
    // 申请统一内存，允许 GPU 和 CPU 访问
    checkCudaErrors(cudaMallocManaged(&data, data_size));

    clock_t start, stop;
    start = clock();

    dim3 blocks((image_width + tx - 1) / tx, (image_width + ty - 1) / ty);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(data, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";

    stbi_write_png("../RayTracing.png", image_width, image_height, channels, data, 0);
    // 释放内存
    checkCudaErrors(cudaFree(data));
    return 0;
}
