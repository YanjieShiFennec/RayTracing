#ifndef BVH_H
#define BVH_H

#include "rt_constants.h"

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

#include <algorithm>

void swap(shared_ptr<hittable*> p1, shared_ptr<hittable*> p2) {
    hittable *temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

void bubble_sort(std::vector<shared_ptr<hittable>> &objects, size_t start, size_t end, int axis_index) {
    for (int i = start; i < end - 1; i++) {
        for (int j = start; j < end - (i - start) - 1; j++) {
            auto a_axis_interval = objects[j]->bounding_box().axis_interval(axis_index);
            auto b_axis_interval = objects[j + 1]->bounding_box().axis_interval(axis_index);

            if (a_axis_interval.min > b_axis_interval.min) {
                // std::swap(objects[j], objects[j + 1]);

                // std::iter_swap(objects.begin() + j, objects.begin() + j + 1);

                // shared_ptr<hittable> temp = objects[j];
                // objects[j] = objects[j + 1];
                // objects[j + 1] = temp;

                swap(objects[j], objects[j + 1]);
            }
        }
    }
}

class bvh_node : public hittable {
public:
    bvh_node(hittable_list list) : bvh_node(list.objects, 0, list.objects.size()) {
        // There's a C++ subtlety here. This constructor (without span indices) creates an
        // implicit copy of the hittable list, which we will modify. The lifetime of the copied
        // list only extends until this constructor exits. That's OK, because we only need to
        // persist the resulting bounding volume hierarchy.
    }

    bvh_node(std::vector<shared_ptr<hittable>> &objects, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb::empty;
        for (size_t object_index = start; object_index < end; object_index++)
            bbox = aabb(bbox, objects[object_index]->bounding_box());

        int axis = bbox.longest_axis();

        auto comparator = (axis == 0) ? box_x_compare
                                      : (axis == 1) ? box_y_compare
                                                    : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);
            // bubble_sort(objects, start, end, axis);

            auto mid = start + object_span / 2;
            left = make_shared<bvh_node>(objects, start, mid);
            right = make_shared<bvh_node>(objects, mid, end);
        }
    }

    bool hit(const ray &r, interval ray_t, hit_record &rec) const override {
        if (!bbox.hit(r, ray_t)) {
            return false;
        }

        /*
         * 递归形式
         */
        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
        /*
         * 递归形式
         */

        /*
         * 非递归形式
         */
        shared_ptr<hittable> stack[64];
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
                if (node->hit(r, interval(ray_t.min, temp_ray_max), rec)) {
                    hit_anything = true;
                    temp_ray_max = rec.t;
                }
            } else {
                // 入栈
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

    void print() const override {
        std::cout << bbox.x.min << " " << bbox.y.min << " " << bbox.z.min << std::endl;
        left->print();
        right->print();
    }

    aabb bounding_box() const override { return bbox; }

    shared_ptr<hittable> get_left_child() const override {
        return left;
    };

    shared_ptr<hittable> get_right_child() const override {
        return right;
    };

private:
    shared_ptr<hittable> left;
    shared_ptr<hittable> right;
    aabb bbox;

    static bool box_compare(
            const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis_index
    ) {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    static bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
        return box_compare(a, b, 2);
    }
};

#endif // BVH_H
