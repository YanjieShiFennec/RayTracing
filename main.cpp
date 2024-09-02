#include "rt_constants.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "quad.h"
#include "bvh.h"
#include "texture.h"

// cmake-build-debug/RayTracing > image.ppm

void bouncing_spheres() {
    // 设置球体
    hittable_list world;

    auto checker = make_shared<checker_texture>(0.32f, color(0.2f, 0.3f, 0.1f), color(0.9f, 0.9f, 0.9f));
    world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, make_shared<lambertian>(checker)));

    // auto material_ground = make_shared<lambertian>(color(0.5f, 0.5f, 0.5f));
    // world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, material_ground));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            point3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8f) {
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

    auto material1 = make_shared<dielectric>(1.5f);
    world.add(make_shared<sphere>(point3(0.0f, 1.0f, 0.0f), 1.0f, material1));

    auto material2 = make_shared<lambertian>(color(0.4f, 0.2f, 0.1f));
    world.add(make_shared<sphere>(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

    auto material3 = make_shared<metal>(color(0.7f, 0.6f, 0.5f), 0.0f);
    world.add(make_shared<sphere>(point3(4.0f, 1.0f, 0.0f), 1.0f, material3));

    // create bvh
    // world.print();
    auto bn = make_shared<bvh_node>(world);
    world = hittable_list(bn);

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
    cam.background = color(0.7f, 0.8f, 1.0f);

    cam.vfov = 20.0f;
    cam.lookfrom = point3(13.0f, 2.0f, 3.0f);
    cam.lookat = point3(0.0f, 0.0f, -1.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.6f;
    cam.focus_dist = 10.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

void checkered_spheres() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(0.32f, color(0.2f, 0.3f, 0.1f), color(0.9f, 0.9f, 0.9f));
    world.add(make_shared<sphere>(point3(0.0f, -10.0f, 0.0f), 10.0f, make_shared<lambertian>(checker)));
    world.add(make_shared<sphere>(point3(0.0f, 10.0f, 0.0f), 10.0f, make_shared<lambertian>(checker)));

    camera cam;

    cam.aspect_ratio = 16.0f / 9.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 10;
    cam.max_depth = 50;
    cam.background = color(0.7f, 0.8f, 1.0f);

    cam.vfov = 20.0f;
    cam.lookfrom = point3(13.0f, 2.0f, 3.0f);
    cam.lookat = point3(0.0f, 0.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

void earth() {
    auto earth_texture = make_shared<image_texture>("earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0.0f, 0.0f, 0.0f), 2.0f, earth_surface);

    camera cam;

    cam.aspect_ratio = 16.0f / 9.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(0.7f, 0.8f, 1.0f);

    cam.vfov = 20.0f;
    cam.lookfrom = point3(0.0f, 0.0f, 12.0f);
    cam.lookat = point3(0.0f, 0.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(hittable_list(globe), file_name);
}

void perlin_spheres() {
    hittable_list world;

    auto pertext = make_shared<noise_texture>(4);
    world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, make_shared<lambertian>(pertext)));
    world.add(make_shared<sphere>(point3(0.0f, 2.0f, 0.0f), 2.0f, make_shared<lambertian>(pertext)));

    camera cam;

    cam.aspect_ratio = 16.0f / 9.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(0.7f, 0.8f, 1.0f);

    cam.vfov = 20.0f;
    cam.lookfrom = point3(13.0f, 2.0f, 3.0f);
    cam.lookat = point3(0.0f, 0.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

void quads() {
    hittable_list world;

    // Materials
    auto left_red = make_shared<lambertian>(color(1.0f, 0.2f, 0.2f));
    auto back_green = make_shared<lambertian>(color(0.2f, 1.0f, 0.2f));
    auto right_blue = make_shared<lambertian>(color(0.2f, 0.2f, 1.0f));
    auto upper_orange = make_shared<lambertian>(color(1.0f, 0.5f, 0.0f));
    auto lower_teal = make_shared<lambertian>(color(0.2f, 0.8f, 0.8f));

    // Quads
    world.add(make_shared<quad>(point3(-3.0f, -2.0f, 5.0f), vec3(0.0f, 0.0f, -4.0f), vec3(0.0f, 4.0f, 0.0f), left_red));
    world.add(
            make_shared<quad>(point3(-2.0f, -2.0f, 0.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 4.0f, 0.0f), back_green));
    world.add(make_shared<quad>(point3(3.0f, -2.0f, 1.0f), vec3(0.0f, 0.0f, 4.0f), vec3(0.0f, 4.0f, 0.0f), right_blue));
    world.add(
            make_shared<quad>(point3(-2.0f, 3.0f, 1.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 4.0f), upper_orange));
    world.add(
            make_shared<quad>(point3(-2.0f, -3.0f, 5.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -4.0f), lower_teal));

    camera cam;

    cam.aspect_ratio = 1.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(0.7f, 0.8f, 1.0f);

    cam.vfov = 80.0f;
    cam.lookfrom = point3(0.0f, 0.0f, 9.0f);
    cam.lookat = point3(0.0f, 0.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

void simple_light() {
    hittable_list world;

    auto per_text = make_shared<noise_texture>(4.0f);
    world.add(make_shared<sphere>(point3(0.0f, -1000.0f, 0.0f), 1000.0f, make_shared<lambertian>(per_text)));
    world.add(make_shared<sphere>(point3(0.0f, 2.0f, 0.0f), 2.0f, make_shared<lambertian>(per_text)));

    // Note that the light is brighter than (1,1,1). This allows it to be bright enough to light things.
    auto diff_light = make_shared<diffuse_light>(color(4.0f, 4.0f, 4.0f));
    world.add(make_shared<quad>(point3(3.0f, 1.0f, -2.0f), vec3(2.0f, 0.0f, 0.0f), vec3(0.0f, 2.0f, 0.0f), diff_light));
    world.add(make_shared<sphere>(point3(0.0f, 7.0f, 0.0f), 2.0f, diff_light));

    camera cam;

    cam.aspect_ratio = 16.0f / 9.0f;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;
    cam.background = color(0.0f, 0.0f, 0.0f);

    cam.vfov = 20.0f;
    cam.lookfrom = point3(26.0f, 3.0f, 6.0f);
    cam.lookat = point3(0.0f, 2.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

void cornell_box() {
    // This image is very noisy because the light is small, so most random rays don't hit the light source.
    hittable_list world;

    auto red = make_shared<lambertian>(color(0.65f, 0.05f, 0.05f));
    auto white = make_shared<lambertian>(color(0.73, 0.73, 0.73));
    auto green = make_shared<lambertian>(color(0.12, 0.45, 0.15));
    auto light = make_shared<diffuse_light>(color(15.0f, 15.0f, 15.0f));

    world.add(make_shared<quad>(point3(555.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), green));
    world.add(make_shared<quad>(point3(0.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), red));
    world.add(make_shared<quad>(point3(343.0f, 554.0f, 332.0f), vec3(-130.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -105.0f),
                                light));
    world.add(make_shared<quad>(point3(0.0f, 0.0f, 0.0f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), white));
    world.add(make_shared<quad>(point3(555.0f, 555.0f, 555.0f), vec3(-555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -555.0f),
                                white));
    world.add(make_shared<quad>(point3(0.0f, 0.0f, 555.0f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.0f), white));

    // two blocks
    // auto b = box(point3(130.0f, 0.0f, 65.0f), point3(295.0f, 165.0f, 230.0f), white);
    // for (int i = 0; i < 6; i++)
    //     world.add(b->objects[i]);
    // b = box(point3(265.0f, 0.0f, 295.0f), point3(430.0f, 330.0f, 460.0f), white);
    // for (int i = 0; i < 6; i++)
    //     world.add(b->objects[i]);

    // world.add(box(point3(130.0f, 0.0f, 65.0f), point3(295.0f, 165.0f, 230.0f), white));
    // world.add(box(point3(265.0f, 0.0f, 295.0f), point3(430.0f, 330.0f, 460.0f), white));

    shared_ptr<hittable> box1 = box(point3(0.0f, 0.0f, 0.0f), point3(165.0f, 330.0f, 165.0f), white);
    box1 = make_shared<rotate_y>(box1, 15.0f);
    box1 = make_shared<translate>(box1, vec3(265.0f, 0.0f, 295.0f));
    world.add(box1);

    shared_ptr<hittable> box2 = box(point3(0.0f, 0.0f, 0.0f), point3(165.0f, 165.0f, 165.0f), white);
    box2 = make_shared<rotate_y>(box2, -18.0f);
    box2 = make_shared<translate>(box2, vec3(130.0f, 0.0f, 65.0f));
    world.add(box2);

    camera cam;

    cam.aspect_ratio = 1.0f;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_depth = 50;
    cam.background = color(0.0f, 0.0f, 0.0f);

    cam.vfov = 40.0f;
    cam.lookfrom = point3(278.0f, 278.0f, -800.0f);
    cam.lookat = point3(278.0f, 278.0f, 0.0f);
    cam.vup = vec3(0.0f, 1.0f, 0.0f);

    cam.defocus_angle = 0.0f;

    char file_name[] = "../RayTracing.png";
    cam.render(world, file_name);
}

int main() {
    switch (7) {
        case 1:
            bouncing_spheres();
            break;
        case 2:
            checkered_spheres();
            break;
        case 3:
            earth();
            break;
        case 4:
            perlin_spheres();
            break;
        case 5:
            quads();
            break;
        case 6:
            simple_light();
            break;
        case 7:
            cornell_box();
            break;
    }
}
