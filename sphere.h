//
// Created by 卢本伟 on 2024/5/30.
//

#ifndef SPHERE_H
#define SPHERE_H

#include "rt_constants.h"

#include "hittable.h"

class sphere : public hittable {
public:
    sphere(const point3 &center, double radius, shared_ptr<material> mat)
    : center(center), radius(fmax(0, radius)),mat(mat) {}

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
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
        // auto a = dot(r.direction(), r.direction());
        // auto b = -2.0 * dot(r.direction(), oc);
        // auto c = dot(oc, oc) - radius * radius;
        // auto discriminant = b * b - 4 * a * c;
        // 将 b = -2h 代入简化求根公式
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;
        auto discriminant = h * h - a * c;
        if (discriminant < 0) {
            return false;
        }

        auto sqrtd = sqrt(discriminant);
        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        // 获取指向球面外的法向量
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

private:
    point3 center;
    double radius;
    shared_ptr<material> mat;
};

#endif // SPHERE_H
