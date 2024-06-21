#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_constants.h"

#include "hittable.h"
#include "aabb.h"

class hittable_list : public hittable {
public:
    hittable **objects;
    int size;
    int allocated_size;

    __device__ hittable_list(): objects(new hittable *[1]), size(0), allocated_size(1) {
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
        bbox = aabb(bbox, object->bounding_box());
    }

    __device__ void remove(int index) {
        if (index < 0 || index >= size)
            return;

        if (index != size - 1) {
            for (int i = index; i < size - 1; i++) {
                objects[i] = objects[i + 1];
            }
        }
        size--;
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

    __device__ aabb bounding_box() const override { return bbox; }

private:
    aabb bbox;
};

#endif // HITTABLE_LIST_H
