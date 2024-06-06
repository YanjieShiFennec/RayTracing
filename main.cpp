#include <time.h>

#include "rt_constants.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

// cmake-build-debug/RayTracing > image.ppm

int main() {
    // 设置球体
    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.0);
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100, material_ground));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 1080;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    // 渲染计时
    clock_t start, end;
    start = clock();

    string file_name = "PPMTest.ppm";
    cam.render(world, file_name);

    end = clock();
    std::cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << " s" << std::endl;

    return 0;
}
