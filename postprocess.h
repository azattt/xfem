#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "levelset.h"

struct PostprocessStress
{
    std::array<int, 4> element;
};

// struct GaussPointStress{
//     Eigen::Vector<2, double> coord;

// };

void drawHeavisideElements(const std::vector<HeavisideEnriched> &heaviside_enriched, const QuadMesh &mesh,
                           const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation,
                           const std::vector<LevelSetSign> &vertices_level_set_signs, const Eigen::VectorXd &u,
                           const std::vector<unsigned int> &node_offset,
                           const std::vector<Eigen::Vector2d> &vertices_displaced);