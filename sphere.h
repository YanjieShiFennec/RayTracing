#ifndef SPHERE_H
#define SPHERE_H

#include "rt_constants.h"

#include "hittable.h"
#include "aabb.h"

class sphere : public hittable {
public:
    // Stationary Sphere
    __device__ sphere(const point3 &center, float radius, material *mat)
        : center1(center), radius(fmax(0.0f, radius)), mat(mat), is_moving(false) {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center1 - rvec, center1 + rvec);
    }

    // Moving sphere
    __device__ sphere(const point3 &center1, const point3 &center2, float radius, material *mat) : center1(center1),
        radius(fmax(0.0f, radius)), mat(mat), is_moving(true) {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(center1 - rvec, center1 + rvec);
        aabb box2(center2 - rvec, center2 + rvec);
        bbox = aabb(box1, box2);

        center_vec = center2 - center1;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
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
        point3 center = is_moving ? sphere_center(r.time()) : center1;
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
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ void print() const override {
        printf("%f %f %f\n", bbox.x.min, bbox.y.min, bbox.z.min);
    }

    __device__ hittable *get_left_child() const override {
        return nullptr;
    };

    __device__ hittable *get_right_child() const override {
        return nullptr;
    };

private:
    point3 center1;
    float radius;
    material *mat;
    bool is_moving;
    vec3 center_vec;
    aabb bbox;

    __device__ point3 sphere_center(float time) const {
        // Linearly interpolate from center1 to center2 according to time, where t = 0 yields
        // center1, and t = 1 yields center2.
        return center1 + time * center_vec;
    }

    __device__ void get_sphere_uv(const point3 &p, float &u, float &v) const{
        // p: a given point on the sphere of radius one, centered at the origin/
        // u: returned value [0, 1] of angle around the Y axis from X = -1.
        // v: returned value [0, 1] of angle from Y = -1 to Y = +1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        // [0, pi]
        auto theta = acosf(-p.y());
        // [0, 2pi]
        auto phi = atan2f(-p.z(), p.x()) + pi;

        // [0, 1]
        u = phi / (2 * pi);
        // [0, 1]
        v = theta / pi;
    }
};

#endif // SPHERE_H
