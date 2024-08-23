#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    ray() {}

    ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction), tm(0.0f) {}

    ray(const point3 &origin, const vec3 &direction, float time) : orig(origin), dir(direction), tm(time) {}

    const point3 &origin() const { return orig; }

    const vec3 &direction() const { return dir; }

    float time() const { return tm; }

    point3 at(float t) const {
        return orig + t * dir;
    }

private:
    // P(t) = Q + td（Q 为 orig，d 为 dir）
    // 根据输入 t 确定最终光线到达的点 P: point3 at(float t)
    point3 orig; // 光源坐标
    vec3 dir; // 光线方向
    float tm; // 动态模糊
};

#endif // RAY_H
