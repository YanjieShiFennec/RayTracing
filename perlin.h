#ifndef PERLIN_H
#define PERLIN_H

#include "rt_constants.h"

class perlin {
public:
    __device__ perlin(curandState &rand_state) {
        for (int i = 0; i < point_count; i++)
            randfloat[i] = random_float(rand_state);

        perlin_generate_perm(perm_x, rand_state);
        perlin_generate_perm(perm_y, rand_state);
        perlin_generate_perm(perm_z, rand_state);
    }

    __device__ float noise(const point3 &p) const {
        auto u = p.x() - floorf(p.x());
        auto v = p.y() - floorf(p.y());
        auto w = p.z() - floorf(p.z());

        // use a Hermite cubic to round off the interpolation
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);

        auto i = int(floorf(p.x()));
        auto j = int(floorf(p.y()));
        auto k = int(floorf(p.z()));

        // 进行三线性插值平滑噪声
        float c[2][2][2];
        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    // x & 255: 将 x 截断成 0 - 255 的整数
                    // ^ 进行异或哈希
                    c[di][dj][dk] = randfloat[perm_x[(i + di) & 255] ^ perm_x[(j + dj) & 255] ^ perm_x[(k + dk) & 255]];

        return trilinear_interp(c, u, v, w);
    }

private:
    static const int point_count = 256;
    float randfloat[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];

    __device__ static void perlin_generate_perm(int *p, curandState &rand_state) {
        for (int i = 0; i < point_count; i++)
            p[i] = i;
        permute(p, point_count, rand_state);
    }

    __device__ static void permute(int *p, int n, curandState &rand_state) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(rand_state, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    __device__ static float trilinear_interp(float c[2][2][2], float u, float v, float w) {
        // 三线性插值
        auto accum = 0.0f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    accum += (i * u + (1 - i) * (1 - u))
                            * (j * v + (1 - j) * (1 - v))
                            * (k * w + (1 - k) * (1 - w))
                            * c[i][j][k];
        return accum;
    }
};

#endif // PERLIN_H
