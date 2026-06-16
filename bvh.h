#pragma once

#include <Eigen/Dense>

struct AABB{
    Eigen::Vector2d p1, p2;
};

struct BVHNode{
    AABB aabb;

    // For more complex implementation
    // int primitive_index;
    // int primitive_count;

    std::vector<unsigned int> triangle_indices;

    int left = -1;
    int right = -1;
};


struct Triangle{
    Eigen::Vector2d v0, v1, v2;
};

struct BVHTree{
    BVHNode root;
    std::vector<Triangle> triangles;
};

AABB computeAABB(const Triangle& triangle) {
    Eigen::Vector2d minVec = triangle.v0.cwiseMin(triangle.v1).cwiseMin(triangle.v2);
    Eigen::Vector2d maxVec = triangle.v0.cwiseMax(triangle.v1).cwiseMax(triangle.v2);
    return {minVec, maxVec};
}

inline double median_x(const Triangle& triangle) noexcept{
    return (triangle.v0.x() + triangle.v1.x() + triangle.v2.x())/3.0;
}

inline double median_y(const Triangle& triangle) noexcept{
    return (triangle.v0.y() + triangle.v1.y() + triangle.v2.y())/3.0;
}

BVHTree constructBVHTree(std::vector<Triangle>& triangles){
    std::vector<AABB> triangles_aabb;
    std::vector<int> primitiveIndices;
    primitiveIndices.reserve(triangles.size());

    for (int i = 0; i < triangles.size(); ++i) {
        primitiveIndices.push_back(i);
    }
    double x_min = std::numeric_limits<double>::infinity(), y_min = std::numeric_limits<double>::infinity(),
     x_max = -std::numeric_limits<double>::infinity(), y_max = -std::numeric_limits<double>::infinity();
    for (const Triangle& triangle: triangles){
        const AABB aabb = computeAABB(triangle);
        triangles_aabb.push_back(computeAABB(triangle));
        x_min = std::min(aabb.p1.x(), x_min);
        y_min = std::min(aabb.p1.y(), y_min);
        x_max = std::max(aabb.p2.x(), x_max);
        y_max = std::max(aabb.p2.y(), y_max);
    }
    AABB root_aabb;
    root_aabb.p1 = Eigen::Vector2d{x_min, y_min};
    root_aabb.p2 = Eigen::Vector2d{x_max, y_max};
    
    
    double median = 0;
    if (triangles.size() > 8){
        Eigen::Vector2d diagonal{root_aabb.p2 - root_aabb.p1};
        if (diagonal.x() > diagonal.y()){
            std::sort(triangles.begin()+0, triangles.begin()+ 0+triangles.size(),
             [](const Triangle& t1, const Triangle& t2){ return median_x(t1) > median_x(t2); });
            for (size_t i = 0; i < triangles.size(); i++){
                median += median_x(triangles[i]);
            }
            median = triangles.size();
        }
    }
}

void processNode(const BVHNode& node, const std::vector<Triangle>& triangles){
    int axis = node.aabb.p2 - node.aabb.p1;
    node.aabb.p2.cwiseMax
    for (size_t i = 0; i < node.triangle_indices; i++){

    }
}

BVHNode treeTraversal(){
    
};

int getClosestPoint(const BVHTree& tree, const Eigen::Vector2d point);


