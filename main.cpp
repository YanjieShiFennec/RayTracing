#define STB_IMAGE_WRITE_IMPLEMENTATION  // 使第三方库 stb_image_write 成为可执行的源码

#include "stb_image_write.h"    // https://github.com/nothings/stb
#include <iostream>
#include <fstream>

// cmake-build-debug/RayTracing > image.ppm

int main() {
    // Image
    int image_width = 256;
    int image_height = 256;

    // Render
    int channels = 3;   // 3通道rgb
    unsigned char *data = new unsigned char[image_width * image_height * channels];
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width - 1);
            auto g = double(j) / (image_height - 1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            data[j * image_width * 3 + 3 * i] = ir;
            data[j * image_width * 3 + 3 * i + 1] = ig;
            data[j * image_width * 3 + 3 * i + 2] = ib;
        }
    }
    stbi_write_png("../RayTracing.png", image_width, image_height, channels, data, 0);
    delete[]data;
    std::clog << "\rDone.\n";
    return 0;
}
