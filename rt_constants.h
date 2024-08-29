#ifndef RT_CONSTANTS_H
#define RT_CONSTANTS_H

#include <cmath>
#include <ctime>
#include <random>
#include <iostream>
#include <limits>
#include <memory>

// C++ Std Usings
using std::make_shared;
using std::shared_ptr;
using std::sqrt;
using std::string;
using std::fabs;
using std::cout;

//Constants
const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

inline float random_float() {
    // Returns a random real in [0, 1).
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min, max).
    return min + (max - min) * random_float();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return int(random_float(min, max + 1));
}
// Common Headers
#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif // RT_CONSTANTS_H
