#ifndef PERLIN_H
#define PERLIN_H

#include "rt_constants.h"

class perlin {
public:
    perlin() {
        for (int i = 0; i < point_count; i++)
            randvec[i] = unit_vector(vec3::random(-1, 1));

        perlin_generate_perm(perm_x);
        perlin_generate_perm(perm_y);
        perlin_generate_perm(perm_z);
    }

    float noise(const point3 &p) const {
        auto u = p.x() - std::floorf(p.x());
        auto v = p.y() - std::floorf(p.y());
        auto w = p.z() - std::floorf(p.z());

        auto i = int(std::floorf(p.x()));
        auto j = int(std::floorf(p.y()));
        auto k = int(std::floorf(p.z()));

        // 进行三线性插值平滑噪声
        vec3 c[2][2][2];
        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    // x & 255: 将 x 截断成 0 - 255 的整数
                    // ^ 进行异或哈希
                    c[di][dj][dk] = randvec[perm_x[(i + di) & 255] ^ perm_x[(j + dj) & 255] ^ perm_x[(k + dk) & 255]];

        return perlin_interp(c, u, v, w);
    }

private:
    static const int point_count = 256;
    float randfloat[point_count];
    vec3 randvec[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];

    static void perlin_generate_perm(int *p) {
        for (int i = 0; i < point_count; i++)
            p[i] = i;
        permute(p, point_count);
    }

    static void permute(int *p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    static float perlin_interp(const vec3 c[2][2][2], float u, float v, float w) {
        // use a Hermite cubic to round off the interpolation
        auto uu = u * u * (3 - 2 * u);
        auto vv = v * v * (3 - 2 * v);
        auto ww = w * w * (3 - 2 * w);

        // 三线性插值
        auto accum = 0.0f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                             * (j * vv + (1 - j) * (1 - vv))
                             * (k * ww + (1 - k) * (1 - ww))
                             * dot(c[i][j][k], weight_v);
                }

        // map accum range from [-1, +1] to [0, 1]
        return 0.5f * (1.0f + accum);
    }
};

#endif // PERLIN_H
