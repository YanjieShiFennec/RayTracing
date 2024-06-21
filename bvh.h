#ifndef BVH_H
#define BVH_H

#include "rt_constants.h"

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {
public:
    __device__ bvh_node(hittable_list list): bvh_node(list.objects, 0, list.size) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
    }

    __device__ bvh_node(hittable **objects, size_t start, size_t end) {
        // To be implemented later.
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ aabb bounding_box() const override { return bbox; };

private:
    hittable *left;
    hittable *right;
    aabb bbox;
};

#endif //BVH_H
