#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image_write.h"    // https://github.com/nothings/stb
#include "rt_constants.h"
#include "rtw_stb_image.h"

#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "quad.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"
#include "texture.h"

#include <curand_kernel.h>

// __global__ 修饰的函数在 GPU 上执行，但是需要在 CPU 端调用
__global__ void render(unsigned char *data, camera **cam, hittable_list **d_world, color *d_color_stack,
                       curandState *rand_state) {
    // CUDA 参数
    // blockId: 块索引, blockDim: 块内的线程数量(tx, ty), threadId: 线程索引, gridDim: 网格内的块数量((image_width + tx - 1) / tx, (image_height + ty - 1) / ty).
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    cam[0]->render(data, d_world, d_color_stack, rand_state, index_x, index_y, stride_x, stride_y);
}

__global__ void create_world_bouncing_spheres(hittable_list **d_world, hittable **d_node, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = rand_state[0];

        d_world[0] = new hittable_list();

        auto checker = new checker_texture(0.32f, color(0.2f, 0.3f, 0.1f), color(0.9f, 0.9f, 0.9f));
        d_world[0]->add(new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(checker)));

        // auto material_ground = new lambertian(color(0.5, 0.5, 0.5));
        // d_world[0]->add(new sphere(point3(0, -1000, 0), 1000, material_ground));

        int n = 11;
        for (int a = -1 * n; a < n; a++) {
            for (int b = -1 * n; b < n; b++) {
                float choose_mat = random_float(local_rand_state);
                point3 center(a + 0.9f * random_float(local_rand_state), 0.2f,
                              b + 0.9f * random_float(local_rand_state));

                if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                    material *sphere_material;

                    if (choose_mat < 0.8f) {
                        // diffuse
                        auto albedo = color::random(local_rand_state) * color::random(local_rand_state);
                        sphere_material = new lambertian(albedo);
                        auto center2 = center + vec3(0.0f, random_float(local_rand_state, 0.0f, 0.5f), 0.0f);
                        d_world[0]->add(new sphere(center, center2, 0.2f, sphere_material));
                    } else if (choose_mat < 0.95f) {
                        // metal
                        auto albedo = color::random(local_rand_state, 0.5f, 1.0f);
                        auto fuzz = random_float(local_rand_state, 0.0f, 0.5f);
                        sphere_material = new metal(albedo, fuzz);
                        d_world[0]->add(new sphere(center, 0.2f, sphere_material));
                    } else {
                        // glass
                        sphere_material = new dielectric(1.5f);
                        d_world[0]->add(new sphere(center, 0.2f, sphere_material));
                    }
                }
            }
        }

        auto material1 = new dielectric(1.5f);
        d_world[0]->add(new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1));

        auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
        d_world[0]->add(new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

        auto material3 = new metal(color(0.7f, 0.6f, 0.5f), 0.0f);
        d_world[0]->add(new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, material3));

        // bvh
        // TODO:比不添加bvh速度慢。。。
        // d_node[0] = new bvh_node(d_world);
        // d_world[0] = new hittable_list(d_node, 1);
    }
}

__global__ void create_world_checkered_spheres(hittable_list **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        auto checker = new checker_texture(0.32f, color(0.2f, 0.3f, 0.1f), color(0.9f, 0.9f, 0.9f));

        d_world[0]->add(new sphere(point3(0.0f, -10.0f, 0.0f), 10.0f, new lambertian(checker)));
        d_world[0]->add(new sphere(point3(0.0f, 10.0f, 0.0f), 10.0f, new lambertian(checker)));
    }
}

__global__ void create_world_earth(hittable_list **d_world, unsigned char *image_texture_data, int width, int height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        auto earth_texture = new image_texture(image_texture_data, width, height);
        auto earth_surface = new lambertian(earth_texture);
        auto globe = new sphere(point3(0.0f, 0.0f, 0.0f), 2.0f, earth_surface);

        d_world[0]->add(globe);
    }
}

__global__ void create_world_perlin_spheres(hittable_list **d_world, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        auto pertext = new noise_texture(4.0f, rand_state[0]);
        d_world[0]->add(new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(pertext)));
        d_world[0]->add(new sphere(point3(0.0f, 2.0f, 0.0f), 2.0f, new lambertian(pertext)));
    }
}

__global__ void create_world_quads(hittable_list **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        // Materials
        auto left_red = new lambertian(color(1.0f, 0.2f, 0.2f));
        auto back_green = new lambertian(color(0.2f, 1.0f, 0.2f));
        auto right_blue = new lambertian(color(0.2f, 0.2f, 1.0f));
        auto upper_orange = new lambertian(color(1.0f, 0.5f, 0.0f));
        auto lower_teal = new lambertian(color(0.2f, 0.8f, 0.8f));

        // Quads
        d_world[0]->add(new quad(point3(-3.0f, -2.0f, 5.0f), vec3(0.0f, 0.0f, -4.0f), vec3(0.0f, 4.0f, 0.0f),
                                 left_red));
        d_world[0]->add(
            new quad(point3(-2.0f, -2.0f, 0.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 4.0f, 0.0f), back_green));
        d_world[0]->add(new quad(point3(3.0f, -2.0f, 1.0f), vec3(0.0f, 0.0f, 4.0f), vec3(0.0f, 4.0f, 0.0f),
                                 right_blue));
        d_world[0]->add(
            new quad(point3(-2.0f, 3.0f, 1.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 4.0f), upper_orange));
        d_world[0]->add(
            new quad(point3(-2.0f, -3.0f, 5.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -4.0f), lower_teal));
    }
}

__global__ void create_world_simple_light(hittable_list **d_world, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        auto per_text = new noise_texture(4.0f, rand_state[0]);
        d_world[0]->add(new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(per_text)));
        d_world[0]->add(new sphere(point3(0.0f, 2.0f, 0.0f), 2.0f, new lambertian(per_text)));

        auto diff_light = new diffuse_light(color(4.0f, 4.0f, 4.0f));
        d_world[0]->add(new sphere(point3(0.0f, 7.0f, 0.0f), 2.0f, diff_light));
        d_world[0]->add(new quad(point3(3.0f, 1.0f, -2.0f), vec3(2.0f, 0.0f, 0.0f), vec3(0.0f, 2.0f, 0.0f),
                                 diff_light));
    }
}

__global__ void create_world_cornell_box(hittable_list **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_world[0] = new hittable_list();

        auto red = new lambertian(color(.65f, .05f, .05f));
        auto white = new lambertian(color(.73f, .73f, .73f));
        auto green = new lambertian(color(.12f, .45f, .15f));
        auto light = new diffuse_light(color(15.0f, 15.0f, 15.0f));

        d_world[0]->add(new quad(point3(555.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f),
                                 green));
        d_world[0]->add(new quad(point3(0.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), red));
        d_world[0]->add(new quad(point3(343.0f, 554.0f, 332.0f), vec3(-130.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -105.0f),
                                 light));
        d_world[0]->add(new quad(point3(0.0f, 0.0f, 0.0f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), white));
        d_world[0]->add(new quad(point3(555.0f, 555.0f, 555.0f), vec3(-555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -555.0f),
                                 white));
        d_world[0]->add(new quad(point3(0.0f, 0.0f, 555.0f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f),
                                 white));

        // two blocks
        hittable *box1 = box(point3(0.0f, 0.0f, 0.0f), point3(165.0f, 330.0f, 165.0f), white);
        box1 = new rotate_y(box1, 15.0f);
        box1 = new translate(box1, vec3(265.0f, 0.0f, 295.0f));
        d_world[0]->add(box1);

        hittable *box2 = box(point3(0.0f, 0.0f, 0.0f), point3(165.0f, 165.0f, 165.0f), white);
        box2 = new rotate_y(box2, -18.0f);
        box2 = new translate(box2, vec3(130.0f, 0.0f, 65.0f));
        d_world[0]->add(box2);
    }
}

__global__ void create_camera(camera **cam, float aspect_ratio, int image_width, int samples_per_pixel,
                              int max_depth, color background,
                              float vfov, point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle,
                              float focus_dist) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cam[0] = new camera(aspect_ratio, image_width, samples_per_pixel, max_depth, background, vfov, lookfrom, lookat,
                            vup,
                            defocus_angle, focus_dist);
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

void process(int choice, float aspect_ratio, int image_width, int samples_per_pixel, int max_depth, color background,
             float vfov,
             point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle, float focus_dist) {
    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    camera **d_cam;
    checkCudaErrors(cudaMalloc(&d_cam, sizeof(camera*)));
    create_camera<<<1, 1>>>(d_cam, aspect_ratio, image_width, samples_per_pixel, max_depth, background, vfov, lookfrom,
                            lookat,
                            vup, defocus_angle, focus_dist);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // CUDA Thread
    int tx = 20; // 线程数量，对应 image_width，视硬件性能而定
    int ty = 20; // 线程数量，对应 image_height，视硬件性能而定

    int grid_dim_x = (image_width + tx - 1) / tx;
    int grid_dim_y = (image_height + ty - 1) / ty;
    dim3 blocks(grid_dim_x, grid_dim_y);
    dim3 threads(tx, ty);

    // Random
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc(&d_rand_state, image_width*image_height*sizeof(curandState)));
    curand_init<<<blocks, threads>>>(d_rand_state, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());

    // Color stack
    color *d_color_stack;
    checkCudaErrors(cudaMalloc(&d_color_stack, image_width*image_height*sizeof(color)*max_depth*2));
    // checkCudaErrors(cudaMalloc(&d_color_stack, grid_dim_x*grid_dim_y*sizeof(color)*max_depth*2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // World
    hittable **d_node;
    checkCudaErrors(cudaMalloc(&d_node, sizeof(hittable*)));
    hittable_list **d_world;
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable_list*)));
    clock_t start, stop;
    start = clock();

    unsigned char *image_texture_data;
    switch (choice) {
        case 1:
            create_world_bouncing_spheres<<<1, 1>>>(d_world, d_node, d_rand_state);
            break;
        case 2:
            create_world_checkered_spheres<<<1, 1>>>(d_world);
            break;
        case 3: {
            int texture_x, texture_y, texture_n;
            unsigned char *image_texture_data_host = stbi_load("../images/earthmap.jpg", &texture_x, &texture_y,
                                                               &texture_n, 0);
            if (image_texture_data_host == nullptr) {
                std::cerr << "file not found!" << std::endl;
            }

            size_t texture_size = texture_x * texture_y * texture_n * sizeof(unsigned char);

            checkCudaErrors(
                cudaMallocManaged(&image_texture_data, texture_size));
            checkCudaErrors(
                cudaMemcpy(image_texture_data, image_texture_data_host, texture_size, cudaMemcpyHostToDevice));

            create_world_earth<<<1,1>>>(d_world, image_texture_data, texture_x, texture_y);
        }
        break;
        case 4:
            create_world_perlin_spheres<<<1, 1>>>(d_world, d_rand_state);
            break;
        case 5:
            create_world_quads<<<1, 1>>>(d_world);
            break;
        case 6:
            create_world_simple_light<<<1, 1>>>(d_world, d_rand_state);
            break;
        case 7:
            // This image is very noisy because the light is small, so most random rays don't hit the light source.
            create_world_cornell_box<<<1, 1>>>(d_world);
            break;
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Create took " << timer_seconds << " seconds.\n";

    // Render
    unsigned char *data;
    size_t data_size = channels * image_width * image_height * sizeof(unsigned char);
    // 申请统一内存，允许 GPU 和 CPU 访问
    checkCudaErrors(cudaMallocManaged(&data, data_size));

    start = clock();

    render<<<blocks, threads>>>(data, d_cam, d_world, d_color_stack, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << timer_seconds << " seconds.\n";

    // 保存为 png
    stbi_write_png(filename, image_width, image_height, channels, data, 0);

    // 释放内存
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_color_stack));
    checkCudaErrors(cudaFree(d_node));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(data));
    if (choice == 3)
        checkCudaErrors(cudaFree(image_texture_data));
}

int main() {
    // Image / Camera params
    float aspect_ratio = 16.0f / 9.0f, vfov = 20.0f, defocus_angle = 0.0f, focus_dist = 10.0f;
    int image_width = 1920, samples_per_pixel = 500, max_depth = 50;
    point3 lookfrom, lookat = point3(0.0f, 0.0f, 0.0f);;
    vec3 vup = vec3(0.0f, 1.0f, 0.0f);
    color background = color(0.7f, 0.8f, 1.0f);

    int choice = 7;
    switch (choice) {
        case 1:
            image_width = 400;
            samples_per_pixel = 50;

            lookfrom = point3(13.0f, 2.0f, 3.0f);
            lookat = point3(0.0f, 0.0f, -1.0f);

            defocus_angle = 0.6f;
            break;
        case 2:
            lookfrom = point3(13.0f, 2.0f, 3.0f);

            defocus_angle = 0.06f;
            break;
        case 3:
            lookfrom = point3(0.0f, 0.0f, 12.0f);
            break;
        case 4:
            lookfrom = point3(13.0f, 2.0f, 3.0f);
            break;
        case 5:
            aspect_ratio = 1.0f;

            vfov = 80.0f;
            lookfrom = point3(0.0f, 0.0f, 9.0f);
            break;
        case 6:
            background = color(0.0f, 0.0f, 0.0f);

            lookfrom = point3(26.0f, 3.0f, 6.0f);
            lookat = point3(0.0f, 2.0f, 0.0f);
            break;
        case 7:
            aspect_ratio = 1.0f;
            image_width = 600;
            background = color(0.0f, 0.0f, 0.0f);

            vfov = 40.0f;
            lookfrom = point3(278.0f, 278.0f, -800.0f);
            lookat = point3(278.0f, 278.0f, 0.0f);
            break;
    }
    process(choice, aspect_ratio, image_width, samples_per_pixel, max_depth, background, vfov, lookfrom, lookat, vup,
            defocus_angle, focus_dist);
    return 0;
}
