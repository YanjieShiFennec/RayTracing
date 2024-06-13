#ifndef RT_CONSTANTS_H
#define RT_CONSTANTS_H

#include <cmath>
#include <iostream>
#include <ctime>
#include <limits>

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

// C++ Std Usings
using std::make_shared;
using std::sqrt;

// Constants
const int channels = 3; // 3通道 rgb
const char filename[] = "../RayTracing.png";

// #define infinity std::numeric_limits<float>::infinity()
__device__ const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions
__host__ inline double degrees_to_radians(double degrees){
    return degrees * pi / 180;
}

// Common Headers
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "interval.h"

#endif // RT_CONSTANTS_H
