#pragma once

#include <Eigen/Dense>

struct AABB{
    Eigen::Vector2d p1, p2;
};

struct BVHNode{
    AABB aabb;

    int primitive_index;
    int primitive_count;

    int left = -1;
    int right = -1;
};


struct Triangle{
    Eigen::Vector2d v0, v1, v2;
};

struct BVHTree{
    BVHNode root;
};


BVHTree constructBVHTree(const std::vector<){

}

BVHNode treeTraversal(){
    
};

int getClosestPoint(const BVHTree& tree, const Eigen::Vector2d point);


