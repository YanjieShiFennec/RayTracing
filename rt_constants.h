#ifndef RT_CONSTANTS_H
#define RT_CONSTANTS_H

#include <iostream>
#include <ctime>
#include <limits>
#include <curand_kernel.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Constants
const int channels = 3; // 3通道 rgb
const char filename[] = "../RayTracing.png";

// #define infinity std::numeric_limits<float>::infinity()
__device__ const float infinity = std::numeric_limits<float>::infinity();
__device__ const float pi = 3.1415926535897932385f;

// Utility Functions
__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float random_float(curandState &rand_state) {
    // Returns a random real in [0, 1).
    return curand_uniform(&rand_state);
}

__device__ inline float random_float(curandState &rand_state, float min, float max) {
    // Returns a random real in [min, max).
    return min + (max - min) * random_float(rand_state);
}

__device__ inline int random_int(curandState &rand_state, int min, int max) {
    // Returns a random integer in [min, max).
    return int(random_float(rand_state, min, max + 1.0f));
}

// Common Headers
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "interval.h"

#endif // RT_CONSTANTS_H
