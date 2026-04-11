#pragma once
#include <vector>
#include <tiny_bvh.h>


// finds which elements are intersected by the crack
// returns the ids of the elements in the quad mesh, enrichment type and intersection point(s)
struct Enriched{
    unsigned int id;
    int type;

    // the number of points is determined by enrichment type
    // also intersection points are located so that
    // first node is the closest to left bottom counterclockwise than second
    tinybvh::bvhvec4 intersection_points[2]; 


};
struct QuadElement{
    unsigned int v0, v1, v2, v3; // counter-clockwise: left bottom, right bottom, right top, left top
};
struct QuadMesh{
    std::vector<tinybvh::bvhvec4> vertices;
    std::vector<QuadElement> elements;
};
struct CrackLine{
    unsigned int v0, v1;
};
struct Crack{
    std::vector<tinybvh::bvhvec2> vertices;
    std::vector<CrackLine> indices;
};
std::vector<Enriched> find_enriched_elements(const QuadMesh& quad_mesh, const Crack& crack);

