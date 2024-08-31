#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_constants.h"

#include "hittable.h"

class hittable_list : public hittable {
public:
    hittable **objects;
    int list_size;   // 已分配的容量
    int allocated_size; // 数组容量上限

    __device__ hittable_list(): objects(new hittable *[1]), list_size(0), allocated_size(1) {
    }

    __device__ hittable_list(hittable **objects, int size): objects(objects), list_size(size), allocated_size(size) {
    }

    __device__ void add(hittable *object) {
        // 数组扩容
        if (allocated_size <= list_size) {
            allocated_size = list_size * 2;
            hittable **new_objects = new hittable *[allocated_size];
            for (int i = 0; i < list_size; i++) {
                new_objects[i] = objects[i];
            }
            objects = new_objects;
        }

        objects[list_size++] = object;
        bbox = aabb(bbox, object->bounding_box());
    }

    // deprecated
    // __device__ void add(hittable_list *object, int size) {
    //     for (int i = 0; i < size; i++)
    //         add(object->objects[i]);
    // }

    __device__ void remove(int index) {
        if (index < 0 || index >= list_size)
            return;

        if (index != list_size - 1) {
            for (int i = index; i < list_size - 1; i++) {
                objects[i] = objects[i + 1];
            }
        }
        list_size--;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < list_size; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ void print() const override {
        for (int i = 0; i < list_size; i++) {
            printf("%f %f %f\n", objects[i]->bounding_box().x.min, objects[i]->bounding_box().y.min,
                   objects[i]->bounding_box().z.min);
        }
    }

    __device__ hittable *get_left_child() const override {
        return nullptr;
    };

    __device__ hittable *get_right_child() const override {
        return nullptr;
    };

private:
    aabb bbox;
};

#endif // HITTABLE_LIST_H
