//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

__device__ void write_color(unsigned char *data, int pixel_index, const color &pixel_color) {
    // Translate the [0, 1] component values to the byte range [0, 255].
    data[pixel_index] = int(255.999 * pixel_color.x());     // r
    data[pixel_index + 1] = int(255.999 * pixel_color.y()); // g
    data[pixel_index + 2] = int(255.999 * pixel_color.z()); // b
}

#endif // COLOR_H
