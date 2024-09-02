#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_constants.h"
#include "aabb.h"

// hittable.h 和 material.h 构成循环依赖，使用前向声明
class material;

class hit_record {
public:
    point3 p; // 光线击中球面的坐标点
    vec3 normal; // 法向量，这里定义其方向总是与光线方向相反
    shared_ptr<material> mat; // 物体材质，前向声明必须使用指针
    float t; // 与光源的距离
    float u; // uv坐标
    float v; // uv坐标
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

    virtual aabb bounding_box() const = 0;

    virtual void print() const {};

    virtual shared_ptr<hittable> get_left_child() const {
        return nullptr;
    }

    virtual shared_ptr<hittable> get_right_child() const {
        return nullptr;
    }
};

// 平移操作
class translate : public hittable {
public:
    translate(shared_ptr<hittable> object, const vec3 &offset) : object(object), offset(offset) {
        bbox = object->bounding_box() + offset;
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Move the ray backwards by the offset
        ray offset_r(r.origin() - offset, r.direction(), r.time());

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!object->hit(offset_r, ray_t, rec))
            return false;

        // Move the intersection point forwards by the offset
        rec.p += offset;

        return true;
    }

    aabb bounding_box() const override {
        return bbox;
    }

private:
    shared_ptr<hittable> object;
    vec3 offset;
    aabb bbox;
};

// 绕 Y 轴逆时针旋转 θ
/*
 * 点 (x, y, z) 绕 Z 轴旋转时，z 不变，只需求 (x', y')
 * 假设逆时针旋转 θ，(x, y) 与 x 轴夹角为 α，则 (x', y') 与 x 轴夹角为 θ + α
 * 有 sin(α) = y / r，cos(α) = x / r
 * 代入 x' = r * cos(θ + α) = r * (cos(θ)cos(α) - sin(θ)sin(α))，
 *      y' = r * sin(θ + α) = r * (sin(θ)cos(α) + cos(θ)sin(α))
 * 得 x' = cos(θ)x - sin(θ)y，
 *    y' = sin(θ)x + cos(θ)y
 *
 * 本项目三维空间坐标系为右手坐标系，
 * 当绕 Y 轴旋转时，(x, y) 映射为 (z, x)
 * 得 z' = cos(θ)z - sin(θ)x，
 *    x' = sin(θ)z + cos(θ)x
 *
 * 当绕 X 轴旋转时，(x, y) 映射为 (y, z)
 * 得 y' = cos(θ)y - sin(θ)z，
 *    z' = sin(θ)y + cos(θ)z
 */
class rotate_y : public hittable {
public:
    rotate_y(shared_ptr<hittable> object, float angle) : object(object) {
        auto radians = degrees_to_radians(angle);
        sin_theta = std::sinf(radians);
        cos_theta = std::cosf(radians);
        bbox = object->bounding_box();

        // 新的外围盒的最小坐标值与最大坐标值，即左下角和右上角
        point3 min(infinity, infinity, infinity);
        point3 max(-infinity, -infinity, -infinity);

        // 遍历被变换物体的包围盒的8个顶点
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    // 算出当前顶点的新的 x, z 坐标值
                    // bbox 逆时针旋转 θ
                    auto newx = cos_theta * x + sin_theta * z;
                    auto newz = -sin_theta * x + cos_theta * z;

                    vec3 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fminf(min[c], tester[c]);
                        max[c] = std::fmaxf(min[c], tester[c]);
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Transform the ray from world space to object space
        // 光线逆时针旋转 -θ
        auto origin = point3(cos_theta * r.origin().x() - sin_theta * r.origin().z(),
                             r.origin().y(),
                             sin_theta * r.origin().x() + cos_theta * r.origin().z());

        auto direction = vec3(cos_theta * r.direction().x() - sin_theta * r.direction().z(),
                              r.direction().y(),
                              sin_theta * r.direction().x() + cos_theta * r.direction().z());

        ray rotated_r(origin, direction, r.time());

        // Determine whether an intersection exists in objects in object space (and if so, where).
        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.
        // 逆时针旋转 θ
        rec.p = point3(cos_theta * rec.p.x() + sin_theta * rec.p.z(),
                       rec.p.y(),
                       -sin_theta * rec.p.x() + cos_theta * rec.p.z());

        // 法线也需要旋转回去
        rec.normal = vec3(cos_theta * rec.normal.x() + sin_theta * rec.normal.z(),
                          rec.normal.y(),
                          -sin_theta * rec.normal.x() + cos_theta * rec.normal.z());

        return true;
    }

    aabb bounding_box() const override { return bbox; }

private:
    shared_ptr<hittable> object;
    float sin_theta;
    float cos_theta;
    aabb bbox;
};

// 分别绕 x, y, z 轴旋转α, β, γ 度
// https://blog.csdn.net/shenquanyue/article/details/103262512
class rotate : public hittable {
public:
    rotate(shared_ptr<hittable> object, float alpha, float beta, float gamma) : object(object) {
        auto alpha_radians = degrees_to_radians(alpha);
        auto beta_radians = degrees_to_radians(beta);
        auto gamma_radians = degrees_to_radians(gamma);
        sin_alpha = std::sinf(alpha_radians);
        cos_alpha = std::cosf(alpha_radians);
        sin_beta = std::sinf(beta_radians);
        cos_beta = std::cosf(beta_radians);
        sin_gamma = std::sinf(gamma_radians);
        cos_gamma = std::cosf(gamma_radians);
        bbox = object->bounding_box();

        // 新的外围盒的最小坐标值与最大坐标值，即左下角和右上角
        point3 min(infinity, infinity, infinity);
        point3 max(-infinity, -infinity, -infinity);

        // 遍历被变换物体的包围盒的8个顶点
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    auto x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    auto y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    auto z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    // 算出当前顶点的新的 x, z 坐标值
                    // bbox 逆时针旋转 θ
                    vec3 tester = rotate_vec_counter_clockwise(x, y, z);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fminf(min[c], tester[c]);
                        max[c] = std::fmaxf(min[c], tester[c]);
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Transform the ray from world space to object space
        // 光线逆时针旋转 -θ
        auto origin = rotate_vec_negative_clockwise(r.origin().x(), r.origin().y(), r.origin().z());

        auto direction = rotate_vec_negative_clockwise(r.direction().x(), r.direction().y(), r.direction().z());

        ray rotated_r(origin, direction, r.time());

        // Determine whether an intersection exists in objects in object space (and if so, where).
        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.
        // 逆时针旋转 θ
        rec.p = rotate_vec_counter_clockwise(rec.p.x(), rec.p.y(), rec.p.z());

        // 法线也需要旋转回去
        rec.normal = rotate_vec_counter_clockwise(rec.normal.x(), rec.normal.y(), rec.normal.z());

        return true;
    }

    aabb bounding_box() const override { return bbox; }

private:
    shared_ptr<hittable> object;
    float sin_alpha;
    float cos_alpha;
    float sin_beta;
    float cos_beta;
    float sin_gamma;
    float cos_gamma;
    aabb bbox;

    float rotate_x(float x, float y, float z) {
        return cos_beta * cos_gamma * x +
               (sin_alpha * sin_beta * cos_gamma - sin_gamma * cos_alpha) * y +
               (sin_beta * cos_alpha * cos_gamma + sin_alpha * sin_gamma) * z;
    }

    float rotate_y(float x, float y, float z) {
        return cos_beta * sin_gamma * x +
               (cos_alpha * cos_gamma + sin_alpha * sin_beta * sin_gamma) * y +
               (-sin_alpha * cos_gamma + sin_gamma * sin_beta * cos_alpha) * z;
    }

    float rotate_z(float x, float y, float z) {
        return -sin_beta * x +
               sin_alpha * cos_beta * y +
               cos_alpha * cos_beta * z;
    }

    // 逆时针
    vec3 rotate_vec_counter_clockwise(float x, float y, float z) const {
        auto vec_x = cos_beta * cos_gamma * x +
                     (sin_alpha * sin_beta * cos_gamma - sin_gamma * cos_alpha) * y +
                     (sin_beta * cos_alpha * cos_gamma + sin_alpha * sin_gamma) * z;
        auto vec_y = cos_beta * sin_gamma * x +
                     (cos_alpha * cos_gamma + sin_alpha * sin_beta * sin_gamma) * y +
                     (-sin_alpha * cos_gamma + sin_gamma * sin_beta * cos_alpha) * z;
        auto vec_z = -sin_beta * x +
                     sin_alpha * cos_beta * y +
                     cos_alpha * cos_beta * z;
        return vec3(vec_x, vec_y, vec_z);
    }

    // 顺时针
    vec3 rotate_vec_negative_clockwise(float x, float y, float z) const {
        auto vec_x = cos_beta * cos_gamma * x +
                     (sin_alpha * sin_beta * cos_gamma + sin_gamma * cos_alpha) * y +
                     (-sin_beta * cos_alpha * cos_gamma + sin_alpha * sin_gamma) * z;
        auto vec_y = -cos_beta * sin_gamma * x +
                     (cos_alpha * cos_gamma - sin_alpha * sin_beta * sin_gamma) * y +
                     (sin_alpha * cos_gamma + sin_gamma * sin_beta * cos_alpha) * z;
        auto vec_z = sin_beta * x -
                     sin_alpha * cos_beta * y +
                     cos_alpha * cos_beta * z;
        return vec3(vec_x, vec_y, vec_z);
    }
};

#endif // HITTABLE_H
