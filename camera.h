#ifndef CAMERA_H
#define CAMERA_H

#include "rt_constants.h"
#include "hittable.h"
#include "material.h"

class camera {
public:
    float aspect_ratio = 1.0f; // Ratio of image width over height
    int image_width = 100; // Rendered image width in pixel count
    int samples_per_pixel = 10; // Count of random samples for each pixel
    int max_depth = 10; // Maximum number of ray bounces into scene

    // 镜头设置参数
    float vfov = 90.0f; // Vertical view angle (field of view) 垂直可视角度
    point3 lookfrom = point3(0, 0, 0); // Point camera is looking from
    point3 lookat = point3(0, 0, -1); // Point camera is looking at
    vec3 vup = vec3(0, 1, 0); // Camera-relative "up" direction

    // 景深效果参数
    float defocus_angle = 0.0f; // Variant angle of rays through each pixel. 0 表示不启用景深效果
    float focus_dist = 10.0f; // Distance from camera lookfrom point to plane of perfect focus

    __device__ camera(float aspect_ratio, int image_width, int samples_per_pixel,
                      int max_depth, float vfov, point3 lookfrom, point3 lookat, vec3 vup, float defocus_angle,
                      float focus_dist): aspect_ratio(aspect_ratio),
                                         image_width(image_width), samples_per_pixel(samples_per_pixel),
                                         max_depth(max_depth), vfov(vfov), lookfrom(lookfrom), lookat(lookat),
                                         vup(vup), defocus_angle(defocus_angle), focus_dist(focus_dist) {
        initialize();
    }

    __device__ void render(unsigned char *data, hittable_list **d_world, curandState *rand_state, int index_x,
                           int index_y, int stride_x, int stride_y) {
        for (int j = index_y; j < image_height; j += stride_y) {
            for (int i = index_x; i < image_width; i += stride_x) {
                int pixel_index = channels * (j * image_width + i);
                color pixel_color(0, 0, 0);
                curandState local_rand_state = rand_state[pixel_index / channels];

                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j, local_rand_state);
                    pixel_color += ray_color(r, max_depth, d_world, local_rand_state);
                }
                write_color(data, pixel_index, pixel_color * pixel_samples_scale);
            }
        }
    }

    __device__ color ray_color(const ray &r, int depth, hittable_list **d_world, curandState &rand_state) {
        ray cur_ray = r;
        color cur_attenuation = color(1.0, 1.0, 1.0);
        // 击中球面的光线，模拟哑光材料漫反射
        for (int i = 0; i < depth; i++) {
            hit_record rec;
            // 光线反射时由于浮点计算误差导致光源结果可能位于球面内部，此时光线第一次击中球体的距离 t 会非常小，设置 interval 0.001 忽略这种情况
            if (d_world[0]->hit(cur_ray, interval(0.001f, infinity), rec)) {
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
                    cur_attenuation = cur_attenuation * attenuation;
                    cur_ray = scattered;
                } else {
                    return color(0, 0, 0);
                }
            } else {
                // 没有击中球面的光线，可理解为背景颜色，颜色根据高度 y 线性渐变
                // -1.0 < y < 1.0
                vec3 unit_direction = unit_vector(r.direction());
                // 0.0 < a < 1.0
                float a = 0.5f * (unit_direction.y() + 1.0f);
                // 线性渐变
                vec3 c = (1.0f - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        // If we've exceeded the ray bounce limit, no more light is gathered.
        return color(0, 0, 0);
    }

    __device__ vec3 sample_square(curandState &local_rand_state) {
        // Returns the vector to a random point in the [-0.5, -0.5]-[+0.5, +0,5] unit square.
        return vec3(random_float(local_rand_state) - 0.5, random_float(local_rand_state) - 0.5, 0);
    }

    __device__ ray get_ray(int i, int j, curandState &local_rand_state) {
        // Construct a camera ray originating from the defocus disk and directed at randomly sampled
        // point around the pixel, location i, j.

        auto offset = sample_square(local_rand_state);
        auto pixel_sample = pixel00_loc
                            + ((i + offset.x()) * pixel_delta_u)
                            + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(local_rand_state);
        auto ray_direction = pixel_sample - ray_origin;
        return ray(ray_origin, ray_direction);
    }

    __device__ point3 defocus_disk_sample(curandState &rand_state) {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk(rand_state);
        return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

private:
    int image_height; // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    point3 camera_center; // Camera center
    point3 pixel00_loc; // Location of pixel (0, 0)
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below
    vec3 u, v, w; // Camera frame basis vectors
    vec3 defocus_disk_u; // Defocus disk horizontal radius
    vec3 defocus_disk_v; // Defocus disk vertical radius

    __device__ void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0f / samples_per_pixel;

        camera_center = lookfrom;

        // Determine viewport dimensions.
        float theta = degrees_to_radians(vfov);
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h * focus_dist; // 视窗高度，数值越大包含的画面范围越大
        float viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the u, v, w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewport_u = viewport_width * u; // Vector across viewport horizontal edge
        auto viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = camera_center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }
};


#endif // CAMERA_H
