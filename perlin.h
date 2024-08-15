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
        auto i = int(4 * p.x()) & 255;
        auto j = int(4 * p.y()) & 255;
        auto k = int(4 * p.z()) & 255;

        return randfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    }

private:
    static const int point_count = 256;
    float randfloat[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];

    __device__  void perlin_generate_perm(int *p, curandState &rand_state) {
        for (int i = 0; i < point_count; i++)
            p[i] = i;
        permute(p, point_count, rand_state);
    }

    __device__  void permute(int *p, int n, curandState &rand_state) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(rand_state, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};

#endif // PERLIN_H
