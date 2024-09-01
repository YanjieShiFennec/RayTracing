#ifndef QUAD_H
#define QUAD_H

#include "rt_constants.h"

#include "hittable.h"
#include "hittable_list.h"

class quad : public hittable {
public:
    quad(const point3 &Q, const point3 &u, const point3 &v, shared_ptr<material> mat)
        : Q(Q), u(u), v(v), mat(mat) {
        // 平面方程：Ax + By + Cz = D，需要确定四个系数 A, B, C, D
        // 令 normal 为 (A, B, C)，代表平面法线，由两个边向量 u, v 叉乘得出
        auto n = cross(u, v);
        normal = unit_vector(n);
        // (x, y, z) 为平面上任意一点，因此可用 Q 代入方程，
        // 得 normal · Q = D
        D = dot(normal, Q);
        w = n / dot(n, n);

        set_bounding_box();
    }

    virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = aabb(Q, Q + u + v);
        auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        /*
        * 由 ray.h 中的 R(t) = P + td 和 normal · Q = D
        * 得 n · (P + td) = D，求解 t，即交点
        * 得 t = (D - n · P) / (n · d)
        */
        auto denominator = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (std::fabsf(denominator) < 1e-8f)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        auto t = (D - dot(normal, r.origin())) / denominator;
        if (!ray_t.contains(t))
            return false;

        /*
        * 设交点为 P，则根据 Q, u, v 构成的平面坐标系有 P = Q + 𝛼u + 𝛽v
        * 求解 𝛼, 𝛽，若 𝛼, 𝛽 都在 [0, 1] 之间，则表示交点在 quad 内，视为击中，否则视为未击中
        * 设 p = P - Q = 𝛼u + 𝛽v，将 p 分别和 u, v 叉乘，
        * 得 u × p = u × (𝛼u + 𝛽v) = u × 𝛼u + u × 𝛽v = 𝛼(u × u) + 𝛽(u × v)
        * 又向量与自己叉乘自 0 ，得 u × p = 𝛽(u × v)，
        * 向量不能直接进行除法，两边同时点乘 n 转换为标量
        * n · (u × p) = n · 𝛽(u × v)
        * 得 𝛽 = n · (u × p) / (n · (u × v))
        * 同理，根据 a × b = - b × a，得 𝛼 = n · (p × v) / (n · (u × v))
        * 令 w = n / (n · (u × v)) = n / (n · n)
        * 得 𝛼 = w · (p × v), 𝛽 = w · (u × p)
        */
        // Determine if the hit point lies within the planar shape using its plane coordinates.
        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    virtual bool is_interior(float a, float b, hit_record &rec) const {
        interval unit_interval = interval(0.0f, 1.0f);
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

    aabb bounding_box() const override { return bbox; }

    void print() const override {
        std::cout << bbox.x.min << " " << bbox.y.min << " " << bbox.z.min << std::endl;
    }

    shared_ptr<hittable> get_left_child() const override {
        return NULL;
    };

    shared_ptr<hittable> get_right_child() const override {
        return NULL;
    };

private:
    // 平行四边形四个顶点分别为 Q, Q + v, Q + u, Q + u + v
    point3 Q; // 基点
    vec3 u, v; // 两个边向量
    vec3 w;

    shared_ptr<material> mat;
    aabb bbox;
    vec3 normal; // 平面法线
    float D;
};

inline shared_ptr<hittable_list> box(const point3 &a, const point3 &b, shared_ptr<material> mat) {
    // Returns the 3D box (six sides) that contains the two opposite vertices a & b.

    auto sides = make_shared<hittable_list>();

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
    auto max = point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

    auto dx = vec3(max.x() - min.x(), 0.0f, 0.0f);
    auto dy = vec3(0.0f, max.y() - min.y(), 0.0f);
    auto dz = vec3(0.0f, 0.0f, max.z() - min.z());

    sides->add(make_shared<quad>(point3(min.x(), min.y(), max.z()), dx, dy, mat)); // front
    sides->add(make_shared<quad>(point3(max.x(), min.y(), max.z()), -dz, dy, mat)); // right
    sides->add(make_shared<quad>(point3(max.x(), min.y(), min.z()), -dx, dy, mat)); // back
    sides->add(make_shared<quad>(point3(min.x(), min.y(), min.z()), dz, dy, mat)); // left
    sides->add(make_shared<quad>(point3(min.x(), max.y(), max.z()), dx, -dz, mat)); // top
    sides->add(make_shared<quad>(point3(min.x(), min.y(), min.z()), dx, dz, mat)); // bottom

    return sides;
}

#endif // QUAD_H
