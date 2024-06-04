//
// Created by LeoWang on 2024/6/4.
//

#ifndef CAMERA_H
#define CAMERA_H

#include "rt_constants.h"

#include "hittable.h"

class camera {
public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int image_width = 100;      // Rendered image width in pixel count

    void render(const hittable &world) {
        initialize();

        // Render
        std::ofstream outfile("../PPMTest.ppm", std::ios_base::out);
        outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        // std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++) {
            std::clog << "\rScan lines remaining: " << (image_height - j) << ' ' << std::flush;
            for (int i = 0; i < image_width; i++) {
                auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
                auto ray_direction = pixel_center - center;
                ray r(center, ray_direction);
                color pixel_color = ray_color(r, world);
                write_color(std::cout, outfile, pixel_color);
            }
        }
        std::clog << "\rDone.\n";
    }

private:
    int image_height;   // Rendered image height
    point3 center;      // Camera center
    point3 pixel00_loc; // Location of pixel (0, 0)
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    color ray_color(const ray &r, const hittable &world) const {
        hit_record rec;
        // 击中球面的光线，根据法向量对相应球体着色
        if (world.hit(r, interval(0, infinity), rec)) {
            // 法向量区间 [-1, 1]，需变换区间至 [0, 1]
            return 0.5 * (rec.normal + color(1, 1, 1));
        }

        // 没有击中球面的光线，可理解为背景颜色，颜色根据高度 y 线性渐变
        // -1.0 < y < 1.0
        vec3 unit_direction = unit_vector(r.direction());
        // 0.0 < a < 1.0
        auto a = 0.5 * (unit_direction.y() + 1.0);
        // 线性渐变
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }
};


#endif // CAMERA_H