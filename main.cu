#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb
#include "rt_constants.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include <curand_kernel.h>

// __global__ 修饰的函数在 GPU 上执行，但是需要在 CPU 端调用
__global__ void render(unsigned char *data, camera **cam, hittable_list **d_world, curandState *rand_state) {
    // CUDA 参数
    // blockId: 块索引, blockDim: 块内的线程数量, threadId: 线程索引, gridDim: 网格内的块数量.
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    cam[0]->render(data, d_world, rand_state, index_x, index_y, stride_x, stride_y);
}

__global__ void create_world(hittable_list **d_world, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = rand_state[0];

        d_world[0] = new hittable_list();

        auto material_ground = new lambertian(color(0.5, 0.5, 0.5));
        d_world[0]->add(new sphere(point3(0, -1000, 0), 1000, material_ground));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = random_float(local_rand_state);
                point3 center(a + 0.9f * random_float(local_rand_state), 0.2,
                              b + 0.9f * random_float(local_rand_state));

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material *sphere_material;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random(local_rand_state) * color::random(local_rand_state);
                        sphere_material = new lambertian(albedo);
                        auto center2 = center + vec3(0.0f, random_float(local_rand_state, 0.0f, 0.5f), 0.0f);
                        d_world[0]->add(new sphere(center, center2, 0.2, sphere_material));
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(local_rand_state, 0.5, 1);
                        auto fuzz = random_float(local_rand_state, 0, 0.5);
                        sphere_material = new metal(albedo, fuzz);
                        d_world[0]->add(new sphere(center, 0.2, sphere_material));
                    } else {
                        // glass
                        sphere_material = new dielectric(1.5);
                        d_world[0]->add(new sphere(center, 0.2, sphere_material));
                    }
                }
            }
        }

        auto material1 = new dielectric(1.5);
        d_world[0]->add(new sphere(point3(0, 1, 0), 1.0, material1));

        auto material2 = new lambertian(color(0.4, 0.2, 0.1));
        d_world[0]->add(new sphere(point3(-4, 1, 0), 1.0, material2));

        auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
        d_world[0]->add(new sphere(point3(4, 1, 0), 1.0, material3));
    }
}

__global__ void create_camera(camera **cam, float aspect_ratio, int image_width, int samples_per_pixel,
                              int max_depth,
                              float vfov, point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle,
                              float focus_dist) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cam[0] = new camera(aspect_ratio, image_width, samples_per_pixel, max_depth, vfov, lookfrom, lookat, vup,
                            defocus_angle, focus_dist);
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
    // Image / Camera params
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 1920;
    int samples_per_pixel = 500;
    int max_depth = 50;

    float vfov = 20.0f;
    point3 lookfrom = point3(13, 2, 3);
    point3 lookat = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);

    float defocus_angle = 0.6f;
    float focus_dist = 10.0f;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    camera **d_cam;
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(camera*)));
    create_camera<<<1, 1>>>(d_cam, aspect_ratio, image_width, samples_per_pixel, max_depth, vfov, lookfrom, lookat,
                            vup, defocus_angle, focus_dist);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // CUDA Thread
    int tx = 8; // 线程数量，对应 image_width
    int ty = 8; // 线程数量，对应 image_height
    dim3 blocks((image_width + tx - 1) / tx, (image_width + ty - 1) / ty);
    dim3 threads(tx, ty);

    // Random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, image_width*image_height*sizeof(curandState)));
    curand_init<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());

    // World
    hittable_list **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable_list*)));
    create_world<<<1, 1>>>(d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    unsigned char *data;
    size_t data_size = channels * image_width * image_height * sizeof(unsigned char);
    // 申请统一内存，允许 GPU 和 CPU 访问
    checkCudaErrors(cudaMallocManaged(&data, data_size));

    clock_t start, stop;
    start = clock();

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
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(data));
    return 0;
}