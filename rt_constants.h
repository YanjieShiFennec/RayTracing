//
// Created by 卢本伟 on 2024/6/3.
//

#ifndef RT_CONSTANTS_H
#define RT_CONSTANTS_H

#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>

// C++ Std Usings
using std::make_shared;
using std::shared_ptr;
using std::sqrt;
using std::string;
using std::fabs;

//Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180;
}

inline double random_double() {
    // Returns a random real in [0, 1).
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min, max).
    return min + (max - min) * random_double();
}

// Common Headers
#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif // RT_CONSTANTS_H
