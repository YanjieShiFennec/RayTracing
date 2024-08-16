#ifndef TEXTURE_H
#define TEXTURE_H

#include "rt_constants.h"
#include "rtw_stb_image.h"

#include "perlin.h"

class texture {
public:
    virtual ~texture() = default;

    virtual color value(float u, float v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
    solid_color(const color &albedo) : albedo(albedo) {}

    solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

    color value(float u, float v, const point3 &p) const override {
        return albedo;
    }

private:
    color albedo;
};

class checker_texture : public texture {
public:
    checker_texture(float scale, shared_ptr<texture> even, shared_ptr<texture> odd) : inv_scale(1.0f / scale),
                                                                                      even(even), odd(odd) {}

    checker_texture(float scale, const color &c1, const color &c2) : checker_texture(scale,
                                                                                     make_shared<solid_color>(c1),
                                                                                     make_shared<solid_color>(c2)) {}

    color value(float u, float v, const point3 &p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    float inv_scale;
    shared_ptr<texture> even;
    shared_ptr<texture> odd;
};

class image_texture : public texture {
public:
    image_texture(const char *filename) : image(filename) {}

    color value(float u, float v, const point3 &p) const override {
        // if we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return color(0, 0, 1);

        // Clamp input texture coordinates to [0, 1] x [1, 0]
        u = interval(0.0f, 1.0f).clamp(u);
        v = 1.0f - interval(0.0f, 1.0f).clamp(v); // Flip V to image coordinates

        auto i = int(u * image.width());
        auto j = int(v * image.height());
        auto pixel = image.pixel_data(i, j);

        auto color_scale = 1.0f / 255.0f;
        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    rtw_image image;
};

class noise_texture : public texture {
public:
    noise_texture(float scale) : scale(scale) {}

    color value(float u, float v, const point3 &p) const override {
        // make color proportional to something like a sine function,
        // and use turbulence to adjust the phase (so it shifts 𝑥 in sin(𝑥)) which makes the stripes undulate.
        // 让颜色与 sin 函数的值成比例, 并使用扰动函数去调整相位(平移了sin(x)中的x)
        return color(0.5, 0.5, 0.5) * (1 + std::sin(scale * p.z() + 10 * noise.turb(p, 7)));
    }

private:
    perlin noise;
    float scale;
};

#endif // TEXTURE_H
