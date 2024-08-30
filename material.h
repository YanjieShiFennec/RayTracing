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

    __device__ virtual color emitted(float u, float v, const point3 &p) const {
        // 非发光体默认返回黑色
        return color(0.0f, 0.0f, 0.0f);
    }
};

// 漫反射材质
class lambertian : public material {
public:
    __device__ lambertian(const color &albedo) : tex(new solid_color(albedo)) {
    }

    __device__ lambertian(texture *tex) : tex(tex) {
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

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = tex->value(rec.u, rec.v, rec.p);
        return true;
    }

private:
    texture *tex;
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
        scattered = ray(rec.p, reflected, r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

private:
    color albedo;
    float fuzz;
};

// 电介质，产生折射
class dielectric : public material {
public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState &rand_state)
    const override {
        attenuation = color(1.0f, 1.0f, 1.0f); // 玻璃表面不吸收光线
        float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f; // 折射还是全反射
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, ri) > random_float(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media.
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        // 模拟菲涅尔现象
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
    }
};

// 发光体
class diffuse_light : public material {
public:
    __device__ diffuse_light(texture *tex): tex(tex) {
    }

    __device__ diffuse_light(const color &emit): tex(new solid_color(emit)) {
    }

    __device__ color emitted(float u, float v, const point3 &p) const override {
        return tex->value(u, v, p);
    }

private:
    texture *tex;
};

#endif // MATERIAL_H
