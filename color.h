#ifndef COLOR_H
#define COLOR_H

#include "rt_constants.h"

#include "interval.h"
#include "vec3.h"

using color = vec3;

inline float linear_to_gamma(float linear_component) {
    if (linear_component > 0)
        return sqrt(linear_component);
    return 0;
}

void write_color(int rgb[3], const color &pixel_color) {
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    // Apply a linear to gama transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0, 1] component values to the byte range [0, 255].
    static const interval intensity(0.000f, 0.999f);
    rgb[0] = int(256 * intensity.clamp(r));
    rgb[1] = int(256 * intensity.clamp(g));
    rgb[2] = int(256 * intensity.clamp(b));
}

#endif // COLOR_H
