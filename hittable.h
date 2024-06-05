//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_constants.h"

// hittable.h 和 material.h 构成循环依赖，使用前向声明
class material;

class hit_record {
public:
    point3 p; // 光线击中球面的坐标点
    vec3 normal; // 法向量，这里定义其方向总是与光线方向相反
    shared_ptr<material> mat; // 物体材质，前向声明必须使用指针
    double t; // 与光源的距离
    bool front_face; // true 代表光源在球面外部

    void set_face_normal(const ray &r, const vec3 &outward_normal) {
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
    virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const = 0;
};

#endif // HITTABLE_H
