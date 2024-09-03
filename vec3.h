#ifndef VEC3_H
#define VEC3_H

#include "rt_constants.h"

class vec3 {
public:
    float e[3];

    vec3() : e{0.0f, 0.0f, 0.0f} {}

    vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    // const 修饰的是函数 x() 。表示该函数的函数体中的操作不会修改当前这个类的类对象。
    float x() const { return e[0]; }

    float y() const { return e[1]; }

    float z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    float operator[](int i) const { return e[i]; }

    float &operator[](int i) { return e[i]; }

    vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3 &operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator*=(const vec3 &v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    vec3 &operator/=(float t) {
        return *this *= 1.0f / t;
    }

    float length() const {
        return sqrt(length_squared());
    }

    float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        float s = 1e-8f;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    };

    static vec3 random() {
        return vec3(random_float(), random_float(), random_float());
    }

    static vec3 random(float min, float max) {
        return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions
inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator/(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

inline vec3 operator/(const vec3 &v, float t) {
    return (1.0f / t) * v;
}

// 向量内积/点乘
inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

// 向量外积
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

// 单位向量
inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_float(-1.0f, 1.0f), random_float(-1.0f, 1.0f), 0.0f);
        if (p.length_squared() < 1)
            return p;
    }
}

inline vec3 random_in_unit_sphere() {
    while (true) {
        auto p = vec3::random(-1.0f, 1.0f);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

// 生成随机方向的单位向量
inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

// 从球体表面向外发射随机方向的光线
inline vec3 random_on_hemisphere(const vec3 &normal) {
    vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0f)
        // 与球体法向量同向
        return on_unit_sphere;
    else
        // 与球体法向量反向，反转
        return -on_unit_sphere;
}

// 镜面反射
inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2.0f * dot(v, n) * n;
}

// 折射
// uv 为入射光线，n 为介质平面法向量，etai_over_etat 为入射介质的折射率与折射介质的折射率之比
inline vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat) {
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif // VEC3_H
