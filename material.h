#ifndef MATERIAL_H
#define MATERIAL_H

#include "rt_constants.h"

class material {
public:
    virtual ~material() = default;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState &rand_state) const {
        return false;
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState &rand_state)
    const override {
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

#endif // MATERIAL_H
