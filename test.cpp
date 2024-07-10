#include <iostream>

class hittable {
public:
    virtual ~hittable() = default;

    virtual int get_value() const = 0;
};

class val : public hittable {
public:
    val(int value) : value(value) {};

    int get_value() const override { return value; }

private:
    int value;
};

void swap(hittable *&p1, hittable *&p2) {
    hittable *temp = p1;
    p1 = p2;
    p2 = temp;
}

void bubble_sort(hittable **objects, size_t start, size_t end) {
    for (int i = start; i < end - 1; i++) {
        for (int j = start; j < end - (i - start) - 1; j++) {
            int v1 = objects[j]->get_value();
            int v2 = objects[j + 1]->get_value();

            if (v1 > v2) {
                swap(objects[j], objects[j + 1]);
                // hittable *t = objects[j];
                // objects[j] = objects[j + 1];
                // objects[j + 1] = t;
            }
        }
    }
}

class bvh_node : public hittable {
public:
    bvh_node(hittable **objects, size_t start, size_t end) {
        size_t object_span = end - start;

        objects[start]->get_value();
        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start + 1];
        } else {
            bubble_sort(objects, start, end);

            auto mid = start + object_span / 2;
            left = new bvh_node(objects, start, mid);
            right = new bvh_node(objects, mid, end);
        }
        value = left->get_value() + right->get_value();
    }

    int get_value() const override { return value; }

private:
    hittable *left;
    hittable *right;
    int value;
};

class hittable_list : public hittable {
public:
    hittable **objects;
    int size;
    int allocated_size;

    hittable_list() : objects(new hittable *[1]), size(0), allocated_size(1) {
    }

    hittable_list(hittable **objects, int size) : objects(objects), size(size), allocated_size(size) {
    }

    void add(hittable *object) {
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

    void remove(int index) {
        if (index < 0 || index >= size)
            return;

        if (index != size - 1) {
            for (int i = index; i < size - 1; i++) {
                objects[i] = objects[i + 1];
            }
        }
        size--;
    }

    int get_value() const override { return size; }
};


int main() {
    int *v = new int[]{5, 4, 3, 2, 1};
    hittable_list **world = (hittable_list **) malloc(sizeof(hittable_list *));
    *world = new hittable_list();


    for (int i = 0; i < 5; i++) {
        (*world)->add(new val(v[i]));
    }

    for (int i = 0; i < 5; i++) {
        std::cout << (*world)->objects[i]->get_value() << " ";
    }
    std::cout << std::endl;

    bubble_sort((*world)->objects, 1, 4);

    for (int i = 0; i < 5; i++) {
        std::cout << (*world)->objects[i]->get_value() << " ";
    }
    std::cout << std::endl;

    free(world);
    delete[]v;
    return 0;
}