#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_constants.h"

#include "hittable.h"

class hittable_list : public hittable {
public:
    hittable **objects;
    int size;
    int allocated_size;

    __device__ hittable_list() {
    }

    __device__ hittable_list(hittable **objects, int size): objects(objects), size(size), allocated_size(size) {
    }

    __device__ void add(hittable *object) {
        if (allocated_size <= size) {
            hittable **new_objects = new hittable *[size * 2];
            for (int i = 0; i < size; i++) {
                new_objects[i] = objects[i];
            }
            objects = new_objects;
            allocated_size = size * 2;
        }
        objects[size++] = object;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif // HITTABLE_LIST_H
