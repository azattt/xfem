#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

// #include <fcpw/fcpw.h>

// finds which elements are intersected by the crack
// returns the ids of the elements in the quad mesh, enrichment type and intersection point(s)
struct HeavisideEnriched{
    int id;
    // the number of points is determined by enrichment type
    // also intersection points are located so that
    // first node is the closest to left bottom counterclockwise than second
    std::array<Eigen::Vector<double, 2>, 2> intersection_points_local_coords;
    std::array<int, 2> intersected_edges;
};
struct TipEnriched{
    int id;
    // the number of points is determined by enrichment type
    // also intersection points are located so that
    // first node is the closest to left bottom counterclockwise than second
    Eigen::Vector<double, 2> intersection_point_local_coords;
    Eigen::Vector<double, 2> tip_point_local_coords;
    unsigned char intersected_edge;
    unsigned char tip_index;
};

// struct QuadElement{
//     std::array<unsigned int, 4> v; // counter-clockwise: left bottom, right bottom, right top, left top
// };
struct QuadMesh{
    std::vector<Eigen::Vector<double, 2>> vertices;
    std::vector<std::array<int, 4>> elements;
};
struct CrackSegment{
    int v0, v1;
};
struct Crack{
    std::vector<Eigen::Vector<double, 2>> vertices;
    std::vector<CrackSegment> indices;
};

struct CrackSegmentPrecomputed{
    Eigen::Vector<double, 2> v0;
    Eigen::Vector<double, 2> dir;
    double l_squared;
};

struct LevelSetSign
{
    int sign;
    int tip;
    unsigned int index;
};

struct LevelSetFields{
    std::vector<LevelSetSign> vertices_level_set_signs;
    std::vector<double> level_set_1_signed_dist;
    std::vector<double> level_set_2_signed_dist;
    std::vector<double> level_set_3_signed_dist;
};

struct EnrichedElements{
    std::vector<int> regular;
    std::vector<HeavisideEnriched> heaviside_enriched;
    std::vector<TipEnriched> tip_enriched;
    std::vector<bool> heaviside_enriched_nodes; // remember: bool in std::vector is actually 1 bit 
    std::vector<bool> tip_enriched_nodes;
};
struct ShapeData
{
    Eigen::Vector4d N;
    Eigen::Matrix<double, 2, 4> dN_xi_eta; // first row for xi, second row for eta
};

struct HeavisideTriangulation{
    int positive_heaviside_triangles_num;
    std::array<std::array<unsigned char, 3>, 4> tri_indices;
};
struct TipTriangulation{
    std::array<std::array<unsigned char, 3>, 5> tri_indices;
};
struct EnrichedElementsTriangulation{
    std::vector<HeavisideTriangulation> heaviside_enriched_triangulation;
    std::vector<TipTriangulation> tip_enriched_triangulation;
};

void find_enriched_elements(const QuadMesh& quad_mesh, const Crack& crack);
LevelSetFields compute_level_set_fields(const QuadMesh& quad_mesh, const Crack& crack);
EnrichedElements find_enriched_elements_by_level_set_fields_simple(const QuadMesh& quad_mesh, const Crack& crack, const LevelSetFields& level_set_fields);
EnrichedElementsTriangulation triangulate_enriched(const QuadMesh& quad_mesh, const EnrichedElements& enriched_elements, const LevelSetFields& level_set_fields);

