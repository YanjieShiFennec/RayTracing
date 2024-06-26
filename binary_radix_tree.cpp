#include <iostream>

class node {
public:
    int value;

    node(int value) : value(value) {}

    virtual void print() = 0;
};

class inner_node : public node {
public:
    node *left;
    node *right;

    inner_node() : node(0) {}

    inner_node(node *left, node *right, int value) : node(value), left(left), right(right) {}

    void print() override {
        std::cout << "Inner" << value << std::endl;
        left->print();
        right->print();
    }
};

class leaf_node : public node {
public:
    leaf_node() : node(0) {}

    leaf_node(int value) : node(value) {}

    void print() override {
        std::cout << "Leaf" << value << std::endl;
    }
};

// 1 代表开始，-1 代表结尾
int delta(const int *data, int size, int i, int j) {
    if (j >= 0 && j < size) {
        unsigned int di = *(data + i);
        unsigned int dj = *(data + j);

        // Handle duplicates by using index as tiebreaker if necessary.
        if (di == dj) {
            return 32 + __builtin_clz(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(j));
        }
        return __builtin_clz(di ^ dj);
    }
    return -1;
}

// Helper function for dividing with rounding up
int div_rounding_up(int x, int y) {
    return (x + y - 1) / y;
}

inner_node *generateHierarchy(int *data, int size) {
    leaf_node *leaf_nodes = new leaf_node[size];
    inner_node *inner_nodes = new inner_node[size - 1];
    // leaf 结点存放数值
    for (int i = 0; i < size; i++)
        leaf_nodes[i].value = data[i];

    for (int i = 0; i < size - 1; i++) {
        // Determine direction of range
        int d = delta(data, size, i, i + 1) < delta(data, size, i, i - 1) ? -1 : 1;

        // Compute upper bound for the length of the range
        int delta_min = delta(data, size, i, i - d);
        int l_max = 2;
        while (delta(data, size, i, i + l_max * d) > delta_min)
            l_max *= 2;

        // Find the other end using binary search
        int l = 0;
        // 二分查找
        for (int t = l_max / 2; t >= 1; t /= 2) {
            if (delta(data, size, i, i + (l + t) * d) > delta_min)
                l += t;
            else
                break;
        }
        int j = i + l * d;

        // Find the split position using binary search
        int delta_node = delta(data, size, i, j);
        int s = 0, q = 1, t;
        // 二分查找
        while (q <= l) {
            t = div_rounding_up(l, q * 2);
            if (delta(data, size, i, i + (s + t) * d) > delta_node) {
                s += t;
            }
            q *= 2;
        }
        int gamma = i + s * d + std::min(d, 0);

        // Output child pointers
        if (std::min(i, j) == gamma)
            inner_nodes[i].left = &leaf_nodes[gamma];
        else
            inner_nodes[i].left = &inner_nodes[gamma];
        if (std::max(i, j) == gamma + 1)
            inner_nodes[i].right = &leaf_nodes[gamma + 1];
        else
            inner_nodes[i].right = &inner_nodes[gamma + 1];
        // inner 结点存放 index
        inner_nodes[i].value = i;
    }

    return &inner_nodes[0];
};

int main() {
    int size = 8;
    int *data = new int[size]{1, 2, 4, 5, 19, 24, 25, 30};
    // for (int i = 0; i < size; i++) {
    //     *(data + i) = i + 1;
    // }

    inner_node *in = generateHierarchy(data, size);
    in->print();

    delete[]data;
    delete in;
    return 0;
}
