//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    ray() {}

    ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    const point3 &origin() const { return orig; }

    const vec3 &direction() const { return dir; }

    point3 at(double t) const {
        return orig + t * dir;
    }

private:
    point3 orig; // 光源坐标
    vec3 dir; // 光线方向
};

#endif // RAY_H
