#ifndef CAMERA_H
#define CAMERA_H

#include "rt_constants.h"
#include "hittable.h"
#include "material.h"

class camera {
public:
    float aspect_ratio = 1.0; // Ratio of image width over height
    int image_width = 100; // Rendered image width in pixel count
    int samples_per_pixel = 10; // Count of random samples for each pixel
    int max_depth = 10; // Maximum number of ray bounces into scene

    __device__ camera(float aspect_ratio, int image_width, int samples_per_pixel,
                      int max_depth): aspect_ratio(aspect_ratio),
                                      image_width(image_width), samples_per_pixel(samples_per_pixel),
                                      max_depth(max_depth) {
        initialize();
    }

    __device__ int get_image_height() const {
        return image_height;
    }

    __device__ float get_pixel_samples_scale() const {
        return pixel_samples_scale;
    }

    __device__ point3 get_camera_center() const {
        return camera_center;
    }

    __device__ point3 get_pixel00_loc() const {
        return pixel00_loc;
    }

    __device__ vec3 get_pixel_delta_u() const {
        return pixel_delta_u;
    }

    __device__ vec3 get_pixel_delta_v() const {
        return pixel_delta_v;
    }

    __device__ static color ray_color(const ray &r, int depth, hittable_list **d_world, curandState &rand_state) {
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

    __device__ static vec3 sample_square(curandState &local_rand_state) {
        // Returns the vector to a random point in the [-0.5, -0.5]-[+0.5, +0,5] unit square.
        return vec3(random_double(local_rand_state) - 0.5, random_double(local_rand_state) - 0.5, 0);
    }

    __device__ static ray get_ray(int i, int j, camera **cam, curandState &local_rand_state) {
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

private:
    int image_height; // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    point3 camera_center; // Camera center
    point3 pixel00_loc; // Location of pixel (0, 0)
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below

    __device__ void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0f / samples_per_pixel;

        camera_center = point3(0, 0, 0);

        // Determine viewport dimensions.
        float focal_length = 1.0;
        float viewport_height = 2.0; // 视窗高度，数值越大包含的画面范围越大
        float viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
};


#endif // CAMERA_H
