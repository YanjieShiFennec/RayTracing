//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef COLOR_H
#define COLOR_H

#include "rt_constants.h"

#include <fstream>
#include "vec3.h"

using color = vec3;

void write_color(std::ostream &out, std::ofstream &outfile, const color &pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Translate the [0, 1] component values to the byte range [0, 255].
    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    // Write out the pixel color components.
    // out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
    outfile << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif // COLOR_H
