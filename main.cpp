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

    // auto R = cos(pi/4);
    //
    // auto material_left = make_shared<lambertian>(color(0,0,1));
    // auto material_right = make_shared<lambertian>(color(1,0,0));
    //
    // world.add(make_shared<sphere>(point3(-R,0,-1),R,material_left));
    // world.add(make_shared<sphere>(point3(R,0,-1),R,material_right));

    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.5);
    auto material_bubble = make_shared<dielectric>(1.0 / 1.5); // 在 material_left 中嵌入 空气泡，material_left 即为空心玻璃球
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100, material_ground));
    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    cam.vfov = 30;
    cam.lookfrom = point3(-2, 2, 1);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 10.0;
    cam.focus_dist = 3.4;

    // 渲染计时
    clock_t start, end;
    start = clock();

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);

    end = clock();
    cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << " s" << std::endl;

    return 0;
}
