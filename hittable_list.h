#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_constants.h"

#include "hittable.h"

class hittable_list : public hittable {
public:
    hittable **objects;
    int size;

    __device__ hittable_list() {
    }

    __device__ hittable_list(hittable **objects, int size): objects(objects), size(size) {
    }

    __device__ void add(hittable *object) {
        objects[size++] = object;
    }

    __device__ bool hit(const ray &r, float ray_tmin, float ray_tmax, hit_record &rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_tmax;

        for (int i = 0; i < size; i++) {
            if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif // HITTABLE_LIST_H
