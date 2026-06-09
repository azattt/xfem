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

struct TipKResult
{
    int tip_index = 0;
    double K_I = 0.0;
    double K_II = 0.0;
    int used_elements = 0;
};


double computeEquivalentK(double KI, double KII);

double computeCrackGrowthAngle(double KI, double KII);

Eigen::Vector2d rotateVector(
    const Eigen::Vector2d& v,
    double angle
);

bool pointInsideDomain(
    const Eigen::Vector2d& p,
    double w,
    double h
);

bool growCrackOneStep(
    Crack& crack,
    const std::vector<TipKResult>& k_results,
    double KIC,
    double da,
    double domain_w,
    double domain_h
);

template <unsigned int NGauss>
std::vector<TipKResult> computeStress(
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

struct SolveResult
{
    Eigen::VectorXd u_solu;

    std::vector<unsigned int> node_offset;
    std::vector<unsigned int> node_ndof;

    LevelSetFields level_set_fields;
    EnrichedElements enriched_elements;
    EnrichedElementsTriangulation enriched_elements_triangulation;

    Eigen::Vector2d crack_tip_1_t;
    Eigen::Vector2d crack_tip_1_n;
    Eigen::Vector2d crack_tip_2_t;
    Eigen::Vector2d crack_tip_2_n;
};

SolveResult solveCurrentCrackConfiguration(
    const QuadMesh& mesh,
    const Crack& crack,
    double young_modulus,
    double poisson_ratio,
    double sigma,
    double thickness,
    double wh,
    double hh
);

void drawVonMisesStressField(
    const std::vector<TipEnriched>& tip_enriched,
    const std::vector<HeavisideEnriched>& heaviside_enriched,
    const QuadMesh& mesh,
    const std::vector<TipTriangulation>& tip_enriched_triangulation,
    const std::vector<HeavisideTriangulation>& heaviside_enriched_triangulation,
    const Eigen::VectorXd& u_solu,
    const std::vector<unsigned int>& node_offset,
    const std::vector<bool>& heaviside_enriched_nodes,
    const std::vector<bool>& tip_enriched_nodes,
    const std::vector<LevelSetSign>& vertices_level_set_signs,
    const Eigen::Vector2d& crack_tip_1_t,
    const Eigen::Vector2d& crack_tip_1_n,
    const Eigen::Vector2d& crack_tip_2_t,
    const Eigen::Vector2d& crack_tip_2_n,
    const Eigen::Matrix3d& D,
    double scale
);

