#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "gui.h"
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
                           const std::vector<bool> &heaviside_enriched_nodes, double scale);

void drawTipElements(
    const std::vector<TipEnriched>& tip_enriched,
    const std::vector<HeavisideEnriched>& heaviside_enriched,
    const QuadMesh& mesh,
    const Eigen::VectorXd& u_solu,
    const std::vector<unsigned int>& node_offset,
    const std::vector<bool>& heaviside_enriched_nodes,
    const std::vector<bool>& tip_enriched_nodes,
    const std::vector<LevelSetSign>& vertices_level_set_signs,
    double scale,
    std::vector<PolygonalChain>& polygonal_chains,
    const Eigen::Vector2d& crack_tip_1_t,
    const Eigen::Vector2d& crack_tip_1_n,
    const Eigen::Vector2d& crack_tip_2_t,
    const Eigen::Vector2d& crack_tip_2_n
);

template <unsigned int NGauss>
void computeStress(
    const std::vector<TipEnriched> &tip_enriched,
    const std::vector<HeavisideEnriched> &heaviside_enriched,
    const QuadMesh &mesh,
    const std::vector<TipTriangulation> &tip_enriched_triangulation,
    const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation,
    const Eigen::VectorXd &u_solu,
    const std::vector<unsigned int> &node_offset,
    const std::vector<bool> &heaviside_enriched_nodes,
    const std::vector<bool> &tip_enriched_nodes,
    const std::vector<LevelSetSign> &vertices_level_set_signs,
    const std::array<std::array<double, 2>, NGauss> &gauss_pts,
    const std::array<double, NGauss> &gauss_wts,
    const Eigen::Vector2d &crack_tip_1_t,
    const Eigen::Vector2d &crack_tip_1_n,
    const Eigen::Vector2d &crack_tip_2_t,
    const Eigen::Vector2d &crack_tip_2_n,
    const Eigen::Matrix3d &D,
    const double young_modulus,
    const double poisson_ratio,
    const double Rin,
    const double Rout
);