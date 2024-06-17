#ifndef MATERIAL_H
#define MATERIAL_H

#include "rt_constants.h"

class material {
public:
    virtual ~material() = default;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                                    curandState &rand_state) const {
        return false;
    }
};

// 漫反射材质
class lambertian : public material {
public:
    __device__ lambertian(const color &albedo) : albedo(albedo) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState &rand_state)
    const override {
        // 随机方向反射
        // auto scatter_direction = random_on_hemisphere(rec.normal, rand_state);
        // 根据 Lambertian distribution 生成反射方向
        auto scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

// 金属材质
class metal : public material {
public:
    __device__ metal(const color &albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState &rand_state)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

private:
    color albedo;
    float fuzz;
};
#endif // MATERIAL_H
