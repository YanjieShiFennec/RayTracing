#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include <iostream>
#include "external/stb_image.h"

int main() {
    const int bytes_per_pixel = 3;
    auto n = bytes_per_pixel;
    int image_width = 0;         // Loaded image width
    int image_height = 0;        // Loaded image height
    float *fdata = nullptr;
    fdata = stbi_loadf("../images/earthmap.jpg", &image_width, &image_height, &n, bytes_per_pixel);

    std::cout << image_width << " " << image_width << std::endl;
    return 0;
}