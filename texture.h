#ifndef TEXTURE_H
#define TEXTURE_H

#include "rt_constants.h"
#include "perlin.h"

class texture {
public:
    virtual ~texture() = default;

    __device__ virtual color value(float u, float v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
    __device__ solid_color(const color &albedo) : albedo(albedo) {
    }

    __device__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {
    }

    __device__ color value(float u, float v, const point3 &p) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    __device__ checker_texture(float scale, texture *even, texture *odd) : inv_scale(1.0f / scale),
                                                                           even(even), odd(odd) {
    }

    __device__ checker_texture(float scale, const color &c1, const color &c2) : checker_texture(scale,
        new solid_color(c1),
        new solid_color(c2)) {
    }

    __device__ color value(float u, float v, const point3 &p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    float inv_scale;
    texture *even;
    texture *odd;
};

class image_texture : public texture {
public:
    __device__ image_texture(unsigned char *data, int width, int height) : data(data), width(width), height(height) {
    }

    __device__ color value(float u, float v, const point3 &p) const override {
        // if we have no texture data, then return solid cyan as a debugging aid.
        if (height <= 0) return color(0, 0, 1);

        // Clamp input texture coordinates to [0, 1] x [1, 0]
        u = interval(0.0f, 1.0f).clamp(u);
        v = 1.0f - interval(0.0f, 1.0f).clamp(v); // Flip V to image coordinates

        auto i = int(u * width);
        auto j = int(v * height);

        auto color_scale = 1.0f / 255.0f;
        int start = 3 * i + 3 * width * j;
        float r = color_scale * data[start];
        float g = color_scale * data[start + 1];
        float b = color_scale * data[start + 2];
        return color(r, g, b);
    }

private:
    unsigned char *data;
    int width, height;
};

class noise_texture : public texture {
public:
    __device__ noise_texture(float scale, curandState &rand_state) : scale(scale), noise(perlin(rand_state)) {
    }

    __device__ color value(float u, float v, const point3 &p) const override {
        return color(1, 1, 1) * noise.noise(scale * p);
    }

private:
    perlin noise;
    float scale;
};

#endif // TEXTURE_H
