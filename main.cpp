#include "color.h"
#include "vec3.h"
#include "ray.h"

#include <iostream>
#include <fstream>

// cmake-build-debug/RayTracing > image.ppm

double hit_sphere(const point3 &center, double radius, const ray &r) {
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
    // auto a = dot(r.direction(), r.direction());
    // auto b = -2.0 * dot(r.direction(), oc);
    // auto c = dot(oc, oc) - radius * radius;
    // auto discriminant = b * b - 4 * a * c;
    // 将 b = -2h 代入简化求根公式
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = h * h - a * c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - sqrt(discriminant)) / a;
    }
}

color ray_color(const ray &r) {
    // 获取光线击中球体的位置 t，球心坐标为 (0,0,-1)，半径为0.5
    auto t = hit_sphere(point3(0, 0, -1), 0.5, r);
    if (t > 0.0) {
        // 单位法向量
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        // 根据球面法向量进行着色
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }

    // 颜色根据高度 y 线性渐变
    // -1.0 < y < 1.0
    vec3 unit_direction = unit_vector(r.direction());
    // 0.0 < a < 1.0
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

int main() {
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width) / image_height);
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
    std::ofstream outfile("../PPMTest.ppm", std::ios_base::out);
    outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    // std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScan lines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            auto ray_direction = pixel_center - camera_center;
            ray r(camera_center, ray_direction);
            color pixel_color = ray_color(r);
            write_color(std::cout, outfile, pixel_color);
        }
    }
    std::clog << "\rDone.\n";
    return 0;
}
