#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb
#include "rt_constants.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <curand_kernel.h>

__device__ color ray_color(const ray &r, hittable_list **d_world) {
    hit_record rec;
    // 击中球面的光线，根据法向量对相应球体着色
    if (d_world[0]->hit(r, interval(0, infinity), rec)) {
        // 法向量区间 [-1, 1]，需变换区间至 [0, 1]
        return 0.5f * (rec.normal + color(1, 1, 1));
    }

    // 没有击中球面的光线，可理解为背景颜色，颜色根据高度 y 线性渐变
    // -1.0 < y < 1.0
    vec3 unit_direction = unit_vector(r.direction());
    // 0.0 < a < 1.0
    float a = 0.5f * (unit_direction.y() + 1.0f);
    // 线性渐变
    return (1.0f - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

__device__ vec3 sample_square(curandState &local_rand_state) {
    // Returns the vector to a random point in the [-0.5, -0.5]-[+0.5, +0,5] unit square.
    return vec3(curand_uniform(&local_rand_state) - 0.5, curand_uniform(&local_rand_state) - 0.5, 0);
}

__device__ ray get_ray(int i, int j, camera **cam, curandState &local_rand_state) {
    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel, location i, j.

    auto offset = sample_square(local_rand_state);
    auto pixel_sample = cam[0]->get_pixel00_loc()
                        + ((i + offset.x()) * cam[0]->get_pixel_delta_u())
                        + ((j + offset.y()) * cam[0]->get_pixel_delta_v());

    auto ray_origin = cam[0]->get_camera_center();
    auto ray_direction = pixel_sample - ray_origin;
    return ray(ray_origin, ray_direction);
}

// __global__ 修饰的函数在 GPU 上执行，但是需要在 CPU 端调用
__global__ void render(unsigned char *data, camera **cam, hittable_list **d_world, curandState *rand_state) {
    // CUDA 参数
    // blockId: 块索引, blockDim: 块内的线程数量, threadId: 线程索引, gridDim: 网格内的块数量.
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int j = index_y; j < cam[0]->get_image_height(); j += stride_y) {
        for (int i = index_x; i < cam[0]->image_width; i += stride_x) {
            int pixel_index = channels * (j * cam[0]->image_width + i);
            color pixel_color(0, 0, 0);
            curandState local_rand_state = rand_state[pixel_index / channels];
            // printf("%d %d\n", cam[0]->samples_per_pixel, cam[0]->get_pixel_samples_scale());

            for (int sample = 0; sample < cam[0]->samples_per_pixel; sample++) {
                ray r = get_ray(i, j, cam, local_rand_state);
                pixel_color += ray_color(r, d_world);
            }
            write_color(data, pixel_index, pixel_color * cam[0]->get_pixel_samples_scale());
        }
    }
}

__global__ void create_world(hittable **d_list, hittable_list **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(point3(0, 0, -1), 0.5);
        d_list[1] = new sphere(point3(0, -100.5, -1), 100);
        d_world[0] = new hittable_list(d_list, 2);
        // d_world[0]->add(new sphere(point3(0, 1, -1), 0.5));
    }
}

__global__ void create_camera(camera **cam, float aspect_ratio, int image_width, int samples_per_pixel) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cam[0] = new camera(aspect_ratio, image_width, samples_per_pixel);
        // printf("%d %f\n", cam[0]->samples_per_pixel, cam[0]->get_pixel_samples_scale());
    }
}

// 初始化随机数
__global__ void curand_init(curandState *rand_state, int image_width, int image_height) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int j = index_y; j < image_height; j += stride_y) {
        for (int i = index_x; i < image_width; i += stride_x) {
            int pixel_index = j * image_width + i;
            // Each thread gets same seed, a different sequence number, no offset
            curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
        }
    }
}

int main() {
    // Image
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
    int samples_per_pixel = 100;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, image_width*image_height*sizeof(curandState)));

    // Camera
    camera **d_cam;
    checkCudaErrors(cudaMallocManaged(&d_cam, sizeof(camera*)));
    create_camera<<<1, 1>>>(d_cam, aspect_ratio, image_width, samples_per_pixel);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // World
    hittable **d_list;
    checkCudaErrors(cudaMallocManaged(&d_list, 2*sizeof(hittable*)));
    hittable_list **d_world;
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(hittable_list*)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    unsigned char *data;
    size_t data_size = channels * image_width * image_height * sizeof(unsigned char);
    // 申请统一内存，允许 GPU 和 CPU 访问
    checkCudaErrors(cudaMallocManaged(&data, data_size));

    clock_t start, stop;
    start = clock();

    // CUDA Thread
    int tx = 8; // 线程数量，对应 image_width
    int ty = 8; // 线程数量，对应 image_height
    dim3 blocks((image_width + tx - 1) / tx, (image_width + ty - 1) / ty);
    dim3 threads(tx, ty);
    curand_init<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(data, d_cam, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";

    // 保存为 png
    stbi_write_png(filename, image_width, image_height, channels, data, 0);

    // 释放内存
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(data));
    return 0;
}
