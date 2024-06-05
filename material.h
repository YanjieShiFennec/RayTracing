//
// Created by 卢本伟 on 2024/6/5.
//

#ifndef MATERIAL_H
#define MATERIAL_H

#include "rt_constants.h"

class hit_record;

class material {
public:
    virtual ~material() = default;

    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
        return false;
    }
};

#endif // MATERIAL_H
