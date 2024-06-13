//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef SPHERE_H
#define SPHERE_H

#include "rt_constants.h"

#include "hittable.h"

class sphere : public hittable {
public:
    __device__ sphere(const point3 &center, float radius) : center(center), radius(fmax(0.0f, radius)) {
    }

    __device__ bool hit(const ray &r, float ray_tmin, float ray_tmax, hit_record &rec) const override {
        /*
         * 球体公式：x^2 + y^2 + z^2 = r^2
         * 设球心坐标为 C = (Cx,Cy,Cz)，球面上一点坐标为 P = (x,y,z)
         * 则 (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2 = r^2
         * 根据向量内积公式 (C-P)·(C-P) = (Cx - x)^2 + (Cy - y)^2 + (Cz - z)^2
         * 得 (C-P)·(C-P) = r^2（·代表向量内积）
         * 由 ray.h 中的 P(t) = Q + td
         * 得 (C - (Q + td))·(C - (Q + td)) = r^2
         * 其中 t 为未知数，展开得 (d·d)t^2 + (-2d·(C-Q))t + (C-Q)·(C-Q) - r^2 = 0
         * 二元一次方程 b^2 - 4ac >= 0 时有解，说明光线击中球体
         */
        vec3 oc = center - r.origin();
        // float a = dot(r.direction(), r.direction());
        // float b = -2.0f * dot(r.direction(), oc);
        // float c = dot(oc, oc) - radius * radius;
        // float discriminant = b * b - 4.0f * a * c;
        // 将 b = -2h 代入简化求根公式
        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc);
        float c = oc.length_squared() - radius * radius;
        float discriminant = h * h - a * c;

        if (discriminant < 0) {
            return false;
        }

        float sqrtd = sqrt(discriminant);
        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if (root <= ray_tmin || root >= ray_tmax) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || root >= ray_tmax) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        // 获取指向球面外的法向量
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }

private:
    point3 center;
    float radius;
};

#endif // SPHERE_H
