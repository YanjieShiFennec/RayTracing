#ifndef TEXTURE_H
#define TEXTURE_H

#include "rt_constants.h"

class texture {
public:
    virtual ~texture() = default;

    __device__ virtual color value(float u, float v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
    __device__ solid_color(const color &albedo) : albedo(albedo) {}

    __device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

    __device__ color value(float u, float v, const point3 &p) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    __device__ checker_texture(float scale, texture* even, texture* odd) : inv_scale(1.0f / scale),
                                                                                        even(even), odd(odd) {}

    __device__ checker_texture(float scale, const color &c1, const color &c2) : checker_texture(scale,
                                                                                     new solid_color(c1),
                                                                                     new solid_color(c2)) {}

    __device__ color value(float u, float v, const point3 &p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    float inv_scale;
    texture* even;
    texture* odd;
};

#endif // TEXTURE_H