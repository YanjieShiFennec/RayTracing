#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_constants.h"
#include "texture.h"

// hittable.h 和 material.h 构成循环依赖，使用前向声明
class material;

class hit_record {
public:
    point3 p; // 光线击中球面的坐标点
    vec3 normal; // 法向量，这里定义其方向总是与光线方向相反
    material *mat; // 物体材质，前向声明必须使用指针
    float t; // 与光源的距离
    float u; // uv坐标
    float v; // uv坐标
    bool front_face; // true 代表光源在球面外部

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal) {
        // outward_normal 指向球面外的法向量，且要求长度等于单位长度
        // Sets the hit record normal vector.
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    // = default，使用默认实现
    virtual ~hittable() = default;

    // = 0，纯虚函数，无实现
    __device__ virtual bool hit(const ray &r, interval ray_t, hit_record &rec) const = 0;

    __device__ virtual aabb bounding_box() const = 0;

    __device__ virtual void print() const {
    };

    __device__ virtual hittable *get_left_child() const {
        return nullptr;
    }

    __device__ virtual hittable *get_right_child() const {
        return nullptr;
    }
};

// 平移操作
class translate : public hittable {
public:
    __device__ translate(hittable *object, const vec3 &offset) : object(object), offset(offset) {
        bbox = object->bounding_box() + offset;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Move the ray backwards by the offset
        ray offset_r(r.origin() - offset, r.direction(), r.time());

        // Determine whether an intersection exists along the offset ray (and if so, where)
        if (!object->hit(offset_r, ray_t, rec))
            return false;

        // Move the intersection point forwards by the offset
        rec.p += offset;

        return true;
    }

    __device__ aabb bounding_box() const override {
        return bbox;
    }

private:
    hittable *object;
    vec3 offset;
    aabb bbox;
};


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
// 绕 Y 轴逆时针旋转 θ
class rotate_y : public hittable {
public:
    __device__ rotate_y(hittable* object, float angle) : object(object) {
        auto radians = degrees_to_radians(angle);
        sin_theta = sinf(radians);
        cos_theta = cosf(radians);
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
                    vec3 tester = rotate_vec_counter_clockwise(vec3(x, y, z));
                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], tester[c]);
                        max[c] = fmaxf(min[c], tester[c]);
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Transform the ray from world space to object space
        // 光线顺时针旋转 θ
        point3 origin = rotate_vec_clockwise(r.origin());
        auto direction = rotate_vec_clockwise(r.direction());
        ray rotated_r(origin, direction, r.time());

        // Determine whether an intersection exists in objects in object space (and if so, where).
        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.
        // 逆时针旋转 θ
        rec.p = rotate_vec_counter_clockwise(rec.p);
        // 法线也需要旋转回去
        rec.normal = rotate_vec_counter_clockwise(rec.normal);

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

private:
    hittable* object;
    float sin_theta;
    float cos_theta;
    aabb bbox;

    // 逆时针
    __device__ vec3 rotate_vec_counter_clockwise(vec3 v) const {
        auto vec_x = cos_theta * v.x() + sin_theta * v.z();
        auto vec_z = -sin_theta * v.x() + cos_theta * v.z();
        return vec3(vec_x, v.y(), vec_z);
    }

    // 顺时针
    __device__ vec3 rotate_vec_clockwise(vec3 v) const {
        auto vec_x = cos_theta * v.x() - sin_theta * v.z();
        auto vec_z = sin_theta * v.x() + cos_theta * v.z();
        return vec3(vec_x, v.y(), vec_z);
    }
};

// 按顺序绕 x, y, z 轴逆时针旋转 α, β, γ 度（注意绕轴顺序不同得到的结果大部分情况是不同的！）
// https://blog.csdn.net/shenquanyue/article/details/103262512
/*
 * 首先绕 X 轴逆时针旋转 α 度
 * x' = x
 * y' = cos(α)y - sin(α)z
 * z' = sin(α)y + cos(α)z
 *
 * 接着绕 Y 轴逆时针旋转 β 度
 * x'' = sin(β)z' + cos(β)x' = sin(β)sin(α)y + sin(β)cos(α)z + cos(β)x
 * y'' = y' = cos(α)y - sin(α)z
 * z'' = cos(β)z' - sin(β)x' = cos(β)sin(α)y + cos(β)cos(α)z - sin(β)x
 *
 * 最后绕 Z 轴逆时针旋转 γ 度
 * x''' = cos(γ)x'' - sin(γ)y'' = cos(γ)sin(β)sin(α)y + cos(γ)sin(β)cos(α)z + cos(γ)cos(β)x - sin(γ)cos(α)y + sin(γ)sin(α)z
 *                              = cos(β)cos(γ)x + (sin(α)sin(β)cos(γ) - sin(γ)cos(α))y + (sin(β)cos(α)cos(γ) + sin(α)sin(γ))z
 * y''' = sin(γ)x'' + cos(γ)y'' = sin(γ)sin(β)sin(α)y + sin(γ)sin(β)cos(α)z + sin(γ)cos(β)x + cos(γ)cos(α)y - cos(γ)sin(α)z
 *                              = sin(γ)cos(β)x + (sin(α)sin(β)sin(γ) + cos(α)cos(γ))y + (sin(β)sin(γ)cos(α) - sin(α)cos(γ))z
 * z''' = z'' =  - sin(β)x + sin(α)cos(β)y + cos(α)cos(β)z
 *
 *
 * 光线反向旋转同理，按顺序绕 z, y, x 轴顺时针旋转 γ, β, α 度
 * x' = cos(γ)x + sin(γ)y
 * y' = -sin(γ)x + cos(γ)y
 * z' = z
 *
 * x'' = -sin(β)z' + cos(β)x' = -sin(β)z + cos(β)cos(γ)x + cos(β)sin(γ)y
 * y'' = y' = -sin(γ)x + cos(γ)y
 * z'' = cos(β)z' + sin(β)x' = cos(β)z + sin(β)cos(γ)x + sin(β)sin(γ)y
 *
 * x''' = x'' = cos(β)cos(γ)x + sin(γ)cos(β)y - sin(β)z
 * y''' = cos(α)y'' + sin(α)z'' = -sin(γ)cos(α)x + cos(α)cos(γ)y + sin(α)cos(β)z + sin(α)sin(β)cos(γ)x + sin(α)sin(β)sin(γ)y
 * 							    = (sin(α)sin(β)cos(γ) - sin(γ)cos(α))x + (sin(α)sin(β)sin(γ) + cos(α)cos(γ))y + sin(α)cos(β)z
 * z''' = -sin(α)y'' + cos(α)z'' = sin(α)sin(γ)x - sin(α)cos(γ)y + cos(α)cos(β)z + sin(β)cos(α)cos(γ)x + sin(β)sin(γ)cos(α)y
 * 							    = (sin(β)cos(α)cos(γ) + sin(α)sin(γ))x + (sin(β)sin(γ)cos(α) - sin(α)cos(γ))y + cos(α)cos(β)z
 */
class rotate_xyz : public hittable {
public:
    __device__ rotate_xyz(hittable* object, float alpha, float beta, float gamma) : object(object) {
        auto alpha_radians = degrees_to_radians(alpha);
        auto beta_radians = degrees_to_radians(beta);
        auto gamma_radians = degrees_to_radians(gamma);
        sin_alpha = sinf(alpha_radians);
        cos_alpha = cosf(alpha_radians);
        sin_beta = sinf(beta_radians);
        cos_beta = cosf(beta_radians);
        sin_gamma = sinf(gamma_radians);
        cos_gamma = cosf(gamma_radians);
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

                    // 算出当前顶点的新的 x, y, z 坐标值
                    // bbox 逆时针旋转 θ
                    vec3 tester = rotate_vec_counter_clockwise(vec3(x, y, z));
                    for (int c = 0; c < 3; c++) {
                        min[c] = fminf(min[c], tester[c]);
                        max[c] = fmaxf(min[c], tester[c]);
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        // Transform the ray from world space to object space
        // 光线顺时针旋转 θ
        point3 origin = rotate_vec_clockwise(r.origin());
        auto direction = rotate_vec_clockwise(r.direction());
        ray rotated_r(origin, direction, r.time());

        // Determine whether an intersection exists in objects in object space (and if so, where).
        if (!object->hit(rotated_r, ray_t, rec))
            return false;

        // Transform the intersection from object space back to world space.
        // 逆时针旋转 θ
        rec.p = rotate_vec_counter_clockwise(rec.p);
        // 法线也需要旋转回去
        rec.normal = rotate_vec_counter_clockwise(rec.normal);

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

private:
    hittable* object;
    float sin_alpha;
    float cos_alpha;
    float sin_beta;
    float cos_beta;
    float sin_gamma;
    float cos_gamma;
    aabb bbox;

    // 逆时针
    __device__ vec3 rotate_vec_counter_clockwise(vec3 v) const {
        auto vec_x = cos_beta * cos_gamma * v.x() +
                     (sin_alpha * sin_beta * cos_gamma - sin_gamma * cos_alpha) * v.y() +
                     (sin_beta * cos_alpha * cos_gamma + sin_alpha * sin_gamma) * v.z();
        auto vec_y = cos_beta * sin_gamma * v.x() +
                     (cos_alpha * cos_gamma + sin_alpha * sin_beta * sin_gamma) * v.y() +
                     (-sin_alpha * cos_gamma + sin_gamma * sin_beta * cos_alpha) * v.z();
        auto vec_z = -sin_beta * v.x() +
                     sin_alpha * cos_beta * v.y() +
                     cos_alpha * cos_beta * v.z();
        return vec3(vec_x, vec_y, vec_z);
    }

    // 顺时针
    __device__ vec3 rotate_vec_clockwise(vec3 v) const {
        auto vec_x = cos_beta * cos_gamma * v.x() +
                     sin_gamma * cos_beta * v.y() -
                     sin_beta * v.z();
        auto vec_y = (sin_alpha * sin_beta * cos_gamma - sin_gamma * cos_alpha) * v.x() +
                     (sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma) * v.y() +
                     sin_alpha * cos_beta * v.z();
        auto vec_z = (sin_beta * cos_alpha * cos_gamma + sin_alpha * sin_gamma) * v.x() +
                     (sin_beta * sin_gamma * cos_alpha - sin_alpha * cos_gamma) * v.y() +
                     cos_alpha * cos_beta * v.z();
        return vec3(vec_x, vec_y, vec_z);
    }
};

#endif // HITTABLE_H
