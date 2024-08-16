#ifndef BVH_H
#define BVH_H

#include "rt_constants.h"

#include "hittable.h"
#include "hittable_list.h"

__device__ void swap(hittable *&p1, hittable *&p2) {
    hittable *temp = p1;
    p1 = p2;
    p2 = temp;
}

__device__ void bubble_sort(hittable **h, size_t start, size_t end, int index) {
    for (int i = start; i < end - 1; i++) {
        for (int j = start; j < end - (i - start) - 1; j++) {
            auto a_axis_interval = h[j]->bounding_box().axis_interval(index);
            auto b_axis_interval = h[j + 1]->bounding_box().axis_interval(index);

            if (a_axis_interval.min > b_axis_interval.min) {
                swap(h[j], h[j + 1]);

                // hittable *t = h[j];
                // h[j] = h[j + 1];
                // h[j + 1] = t;
            }
        }
    }
}

class bvh_node : public hittable {
public:
    __device__ bvh_node(hittable_list **list): bvh_node(
        list[0]->objects, 0, list[0]->size) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
    }

    __device__ bvh_node(hittable **objects, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb::empty;
        for (size_t object_index = start; object_index < end; object_index++)
            bbox = aabb(bbox, objects[object_index]->bounding_box());

        int axis = bbox.longest_axis();
        // printf("%d\n",axis);
        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            bubble_sort(objects, start, end, axis);

            auto mid = start + object_span / 2;
            left = new bvh_node(objects, start, mid);
            right = new bvh_node(objects, mid, end);
        }
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        /*
         * 递归形式
         */
        // bool hit_left = left->hit(r, ray_t, rec);
        // bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);
        //
        // return hit_left || hit_right;
        /*
         * 递归形式
         */

        /*
         * 非递归形式
         */
        hittable *stack[64];
        int stack_index = 0;
        bool hit_anything = false;
        // 保存光线击中距离避免出现相对位置靠后的 object 覆盖住靠前的 object
        float temp_ray_max = ray_t.max;

        stack[stack_index] = get_left_child();
        stack_index++;
        stack[stack_index] = get_right_child();

        while (stack_index >= 0) {
            // 出栈
            auto node = stack[stack_index];
            stack_index--;

            if (!node->bounding_box().hit(r, ray_t))
                continue;

            if (node->get_left_child() == nullptr) {
                // 叶子结点，满二叉树判断一个即可
                if (node->hit(r, interval(ray_t.min, temp_ray_max), rec)) {
                    hit_anything = true;
                    temp_ray_max = rec.t;
                }
            } else {
                // 非叶子结点则子结点入栈
                stack_index++;
                stack[stack_index] = node->get_left_child();

                stack_index++;
                stack[stack_index] = node->get_right_child();
            }
        }

        return hit_anything;
        /*
         * 非递归形式
         */
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ void print() const override {
        printf("%f %f %f\n", bbox.x.min, bbox.y.min, bbox.z.min);
        left->print();
        right->print();
    }

    __device__ hittable *get_left_child() const override {
        return left;
    };

    __device__ hittable *get_right_child() const override {
        return right;
    };

private:
    hittable *left;
    hittable *right;
    aabb bbox;
};

#endif //BVH_H
