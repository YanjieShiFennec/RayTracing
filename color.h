#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include "interval.h"

using color = vec3;

__device__ inline float linear_to_gamma(float linear_component) {
    if (linear_component > 0)
        return sqrtf(linear_component);
    return 0;
}

__device__ void write_color(unsigned char *data, int pixel_index, const color &pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Apply a linear to gama transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0, 1] component values to the byte range [0, 255].
    interval intensity(0.000, 0.999);
    data[pixel_index] = int(256 * intensity.clamp(r));
    data[pixel_index + 1] = int(256 * intensity.clamp(g));
    data[pixel_index + 2] = int(256 * intensity.clamp(b));
}

#endif // COLOR_H
