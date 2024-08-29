#ifndef CAMERA_H
#define CAMERA_H

#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb

#include "rt_constants.h"

#include "hittable.h"
#include "material.h"

class camera {
public:
    float aspect_ratio = 1.0f;  // Ratio of image width over height
    int image_width = 100;      // Rendered image width in pixel count
    int samples_per_pixel = 10; // Count of random samples for each pixel
    int max_depth = 10;         // Maximum number of ray bounces into
    color background;           // Scene background color

    // 镜头设置参数
    float vfov = 90.0f;           // Vertical view angle (field of view) 垂直可视角度
    point3 lookfrom = point3(0, 0, 0);    // Point camera is looking from
    point3 lookat = point3(0, 0, -1);     // Point camera is looking at
    vec3 vup = vec3(0, 1, 0);             // Camera-relative "up" direction

    // 景深效果参数
    float defocus_angle = 0.0f;   // Variant angle of rays through each pixel. 0 表示不启用景深效果
    float focus_dist = 10.0f;     // Distance from camera lookfrom point to plane of perfect focus

    void render(const hittable &world, const char *file_name) {
        initialize();

        // Render
        int channels = 3;   // 3通道rgb
        unsigned char *data = new unsigned char[image_width * image_height * channels];
        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScan lines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0, 0, 0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }

                int rgb[3];
                write_color(rgb, pixel_samples_scale * pixel_color);
                data[j * image_width * 3 + 3 * i] = rgb[0];
                data[j * image_width * 3 + 3 * i + 1] = rgb[1];
                data[j * image_width * 3 + 3 * i + 2] = rgb[2];
            }
        }
        stbi_write_png(file_name, image_width, image_height, channels, data, 0);
        delete[]data;
        std::clog << "\rDone.\n";
    }

private:
    int image_height;   // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    point3 center;      // Camera center
    point3 pixel00_loc; // Location of pixel (0, 0)
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below
    vec3 u, v, w;       // Camera frame basis vectors
    vec3 defocus_disk_u;// Defocus disk horizontal radius
    vec3 defocus_disk_v;// Defocus disk vertical radius

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0f / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        float theta = degrees_to_radians(vfov);
        float h = tan(theta / 2.0f);
        float viewport_height = 2.0f * h * focus_dist; // 视窗高度，数值越大包含的画面范围越大
        float viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the u, v, w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewport_u = viewport_width * u;   // Vector across viewport horizontal edge
        auto viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2.0f - viewport_v / 2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        float defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the defocus disk and directed at randomly sampled
        // point around the pixel, location i, j.
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                            + ((i + offset.x()) * pixel_delta_u)
                            + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;
        float ray_time = random_float();

        return ray(ray_origin, ray_direction, ray_time);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-0.5, -0.5]-[+0.5, +0,5] unit square.
        return vec3(random_float() - 0.5f, random_float() - 0.5f, 0);
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray &r, int depth, const hittable &world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0, 0, 0);

        hit_record rec;

        // If the ray hits nothing, return the background color.
        // 光线反射时由于浮点计算误差导致光源结果可能位于球面内部，此时光线第一次击中球体的距离 t 会非常小，设置 interval 0.001 忽略这种情况
        if (!world.hit(r, interval(0.001f, infinity), rec)) {
            /*
            // 返回线性渐变
            // 没有击中球面的光线，可理解为背景颜色，颜色根据高度 y 线性渐变
            // -1.0 < y < 1.0
            vec3 unit_direction = unit_vector(r.direction());
            // 0.0 < a < 1.0
            float a = 0.5f * (unit_direction.y() + 1.0f);
            // 线性渐变
            return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
             */

            // 返回固定颜色
            return background;
        }

        ray scattered;
        color attenuation;
        color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

        if (!rec.mat->scatter(r, rec, attenuation, scattered))
            return color_from_emission;

        color color_from_scatter = attenuation * ray_color(scattered, depth - 1, world);

        return color_from_emission + color_from_scatter;
    }
};


#endif // CAMERA_H
