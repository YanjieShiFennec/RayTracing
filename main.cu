#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb
#include <iostream>
#include <ctime>
#include "vec3.h"
#include "color.h"
#include "ray.h"

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

const int channels = 3; // 3通道 rgb
const char filename[] = "../RayTracing.png";

__device__ bool hit_sphere(const point3 &center, float radius, const ray &r) {
    /*
     * 球体公式：x^2 + y^2 + z^2 = r^2
     * 设球心坐标为 C = (Cx,Cy,Cz)，球面上一点坐标为 P = (x,y,z)
     * 则 (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2 = r^2
     * 根据向量内积公式 (C-P)·(C-P) = (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2
     * 得 (C-P)·(C-P) = r^2（·代表向量内积）
     * 由 ray.h 中的 P(t) = Q + td
     * 得 (C - (Q + td))·(C - (Q + td)) = r^2
     * 其中 t 为未知数，展开得 (d·d)t^2 + (-2d·(C-Q))t + (C-Q)·(C-Q) - r^2 = 0
     * 二元一次方程 b^2 - 4ac >= 0 时有解，说明光线击中球体
     */
    vec3 oc = center - r.origin();
    float a = dot(r.direction(), r.direction());
    float b = -2.0f * dot(r.direction(), oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant >= 0);
}

__device__ color ray_color(const ray &r) {
    // 判断光线是否击中球体，球心坐标为 (0,0,-1)，半径为0.5
    if (hit_sphere(point3(0, 0, -1), 0.5, r))
        return color(1, 1, 0);

    // 颜色根据高度 y 线性渐变
    // -1.0 < y < 1.0
    vec3 unit_direction = unit_vector(r.direction());
    // 0.0 < a < 1.0
    float a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

// __global__ 修饰的函数在 GPU 上执行，但是需要在 CPU 端调用
__global__ void render(unsigned char *data, int image_width, int image_height,
                       point3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, point3 camera_center) {
    // CUDA 参数
    // blockId: 块索引, blockDim: 块内的线程数量, threadId: 线程索引, gridDim: 网格内的块数量.
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int j = index_y; j < image_height; j += stride_y) {
        for (int i = index_x; i < image_width; i += stride_x) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);
            color pixel_color = ray_color(r);
            // auto pixel_color = color(float(i) / (image_width - 1), float(j) / (image_height - 1), 0.0);
            int pixel_index = channels * (j * image_width + i);
            write_color(data, pixel_index, pixel_color);
        }
    }
}

int main() {
    // Image
    float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    float focal_length = 1.0;
    float viewport_height = 2.0;
    float viewport_width = viewport_height * (float(image_width) / image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

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
    render<<<blocks, threads>>>(data, image_width, image_height, pixel00_loc, pixel_delta_u, pixel_delta_v,
                                camera_center);
    checkCudaErrors(cudaGetLastError());
    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";

    stbi_write_png(filename, image_width, image_height, channels, data, 0);
    // 释放内存
    checkCudaErrors(cudaFree(data));
    return 0;
}
