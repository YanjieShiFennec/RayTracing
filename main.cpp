#include "rt_constants.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "bvh.h"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
inline unsigned int expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
inline unsigned int morton_3d(point3 centroid) {
    float x = centroid.x();
    float y = centroid.y();
    float z = centroid.z();
    x = fmin(fmax(x * 1024.0f, 0.0f), 1023.0f);
    y = fmin(fmax(y * 1024.0f, 0.0f), 1023.0f);
    z = fmin(fmax(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int) x);
    unsigned int yy = expand_bits((unsigned int) y);
    unsigned int zz = expand_bits((unsigned int) z);
    return xx * 4 + yy * 2 + zz;
}

int main() {
    // 设置球体
    hittable_list world;

    auto material_ground = make_shared<lambertian>(color(0.5f, 0.5f, 0.5f));
    world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, material_ground));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            point3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0.0f, random_float(0.0f, 0.5f), 0.0f);
                    world.add(make_shared<sphere>(center, center2, 0.2f, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5f, 1.0f);
                    float fuzz = random_float(0.0f, 0.5f);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5f);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    size_t size = world.objects.size();
    unsigned int *morton_codes = new unsigned int[size];
    int i = 0;
    for (const auto &object: world.objects) {
        auto centroid = object->bounding_box().get_centroid();
        *(morton_codes + i) = morton_3d(centroid);
        cout << *(morton_codes + i) << std::endl;
        i++;
    }

    delete[]morton_codes;

    world = hittable_list(make_shared<bvh_node>(world));

    // auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    // auto material_left = make_shared<dielectric>(1.5);
    // auto material_bubble = make_shared<dielectric>(1.0 / 1.5); // 在 material_left 中嵌入 空气泡，material_left 即为空心玻璃球
    // auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
    //
    // world.add(make_shared<sphere>(point3(0.0, 0.0, -1.2), 0.5, material_center));
    // world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    // world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    // world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    camera cam;

    cam.aspect_ratio = 16.0f / 9.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 10;
    cam.max_depth = 50;

    cam.vfov = 20.0f;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6f;
    cam.focus_dist = 10.0f;

    // 渲染计时
    clock_t start, end;
    start = clock();

    char file_name[] = "../RayTracing.png";
    // cam.render(world, file_name);

    end = clock();
    cout << "Time: " << double(end - start) / CLOCKS_PER_SEC << " s" << std::endl;

    return 0;
}
