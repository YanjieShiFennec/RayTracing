#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    __device__ ray() {}

    __device__ ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    __device__ const point3 &origin() const { return orig; }

    __device__ const vec3 &direction() const { return dir; }

    __device__ point3 at(float t) const {
        return orig + t * dir;
    }

private:
    // P(t) = Q + td（Q 为 orig，d 为 dir）
    // 根据输入 t 确定最终光线到达的点 P
    point3 orig; // 光源坐标
    vec3 dir; // 光线方向
};

#endif // RAY_H
