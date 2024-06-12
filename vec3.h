#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {
public:
    float e[3];

    // __host__ 修饰的函数在 CPU 端调用，在 CPU 上执行
    // __device__ 修饰的函数在 GPU 端调用，在 GPU 上执行
    __host__ __device__ vec3() : e{0, 0, 0} {}

    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    // const 修饰的是函数 x() 。表示该函数的函数体中的操作不会修改当前这个类的类对象。
    __host__ __device__ float x() const { return e[0]; }

    __host__ __device__ float y() const { return e[1]; }

    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ float operator[](int i) const { return e[i]; }

    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

// point3 is just an alas for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions
inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) {
    return (1 / t) * v;
}

// 向量内积/点乘
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

// 向量外积
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

// 单位向量
__host__ __device__ inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

#endif // VEC3_H
