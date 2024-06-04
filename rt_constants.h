//
// Created by 卢本伟 on 2024/6/3.
//

#ifndef RT_CONSTANTS_H
#define RT_CONSTANTS_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>

// C++ Std Usings
using std::make_shared;
using std::shared_ptr;
using std::sqrt;

//Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees){
    return degrees * pi / 180;
}

// Common Headers
#include "color.h"
#include "ray.h"
#include "vec3.h"

#endif // RT_CONSTANTS_H
