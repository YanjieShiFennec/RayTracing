#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_constants.h"

class hit_record {
public:
    point3 p; // 光线击中球面的坐标点
    vec3 normal; // 法向量，这里定义其方向总是与光线方向相反
    float t; // 与光源的距离
    bool front_face; // true 代表光源在球面外部

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal) {
        // outward_normal 指向球面外的法向量，且要求长度等于单位长度
        // Sets the hit record normal vector.
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    // = default，使用默认实现
    virtual ~hittable() = default;

    // = 0，纯虚函数，无实现
    __device__ virtual bool hit(const ray &r, float ray_tmin, float ray_tmax, hit_record &rec) const = 0;
};

#endif // HITTABLE_H
