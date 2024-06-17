#ifndef MATERIAL_H
#define MATERIAL_H

#include "rt_constants.h"

class material {
public:
    virtual ~material() = default;

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
        return false;
    }
};

#endif // MATERIAL_H
