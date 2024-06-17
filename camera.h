#ifndef CAMERA_H
#define CAMERA_H

#include "rt_constants.h"

#include "hittable.h"

class camera {
public:
    float aspect_ratio = 1.0; // Ratio of image width over height
    int image_width = 100; // Rendered image width in pixel count
    int samples_per_pixel = 10; // Count of random samples for each pixel

    __device__ camera(float aspect_ratio, int image_width, int samples_per_pixel): aspect_ratio(aspect_ratio),
        image_width(image_width), samples_per_pixel(samples_per_pixel) {
        initialize();
    }

    __device__ int get_image_height() const {
        return image_height;
    }

    __device__ float get_pixel_samples_scale() const {
        return pixel_samples_scale;
    }

    __device__ point3 get_camera_center() const {
        return camera_center;
    }

    __device__ point3 get_pixel00_loc() const {
        return pixel00_loc;
    }

    __device__ vec3 get_pixel_delta_u() const {
        return pixel_delta_u;
    }

    __device__ vec3 get_pixel_delta_v() const {
        return pixel_delta_v;
    }

private:
    int image_height; // Rendered image height
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    point3 camera_center; // Camera center
    point3 pixel00_loc; // Location of pixel (0, 0)
    vec3 pixel_delta_u; // Offset to pixel to the right
    vec3 pixel_delta_v; // Offset to pixel below

    __device__ void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0f / samples_per_pixel;

        camera_center = point3(0, 0, 0);

        // Determine viewport dimensions.
        float focal_length = 1.0;
        float viewport_height = 2.0;
        float viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
};


#endif // CAMERA_H
