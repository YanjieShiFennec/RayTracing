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

    point3 at(float t) const {
        return orig + t * dir;
    }

private:
    // P(t) = Q + td（Q 为 orig，d 为 dir）
    // 根据输入 t 确定最终光线到达的点 P: point3 at(float t)
    point3 orig; // 光源坐标
    vec3 dir; // 光线方向
};

#endif // RAY_H
