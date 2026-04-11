#pragma once

#include <iostream>
#include <vector>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>


using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

// finds which elements are intersected by the crack
// returns the ids of the elements in the quad mesh, enrichment type and intersection point(s)
struct Enriched{
    unsigned int id;
    int type;

    // the number of points is determined by enrichment type
    // also intersection points are located so that
    // first node is the closest to left bottom counterclockwise than second
    Vec3 intersection_points[2]; 


};
struct QuadElement{
    unsigned int v0, v1, v2, v3; // counter-clockwise: left bottom, right bottom, right top, left top
};
struct QuadMesh{
    std::vector<Vec3> vertices;
    std::vector<QuadElement> elements;
};
struct CrackLine{
    unsigned int v0, v1;
};
struct Crack{
    std::vector<Vec3> vertices;
    std::vector<CrackLine> indices;
};
std::vector<Enriched> find_enriched_elements(const QuadMesh& quad_mesh, const Crack& crack);

