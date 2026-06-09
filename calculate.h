#pragma once

#include <Eigen/Dense>

#include <vector>

#include "fem.h"
#include "levelset.h"

struct XFemIterationResult
{
    LevelSetFields level_set_fields;
    EnrichedElements enriched_elements;
    EnrichedElementsTriangulation enriched_elements_triangulation;

    std::vector<unsigned int> node_offset;
    std::vector<unsigned int> node_ndof;
    std::vector<bool> active;

    Eigen::VectorXd u;

    Eigen::Vector2d crack_tip_1_t = Eigen::Vector2d::UnitX();
    Eigen::Vector2d crack_tip_1_n = Eigen::Vector2d::UnitY();
    Eigen::Vector2d crack_tip_2_t = Eigen::Vector2d::UnitX();
    Eigen::Vector2d crack_tip_2_n = Eigen::Vector2d::UnitY();

    unsigned int dof_counter = 0;
    double energy = 0.0;
    double residual_norm = 0.0;
};


void computeCrackTipDirections(
    const Crack& crack,
    Eigen::Vector2d& crack_tip_1_t,
    Eigen::Vector2d& crack_tip_1_n,
    Eigen::Vector2d& crack_tip_2_t,
    Eigen::Vector2d& crack_tip_2_n
);

XFemIterationResult solveCrackIteration(
    const QuadMesh& mesh,
    const Crack& crack,
    double w,
    double h,
    int wn,
    int hn,
    double thickness,
    const Eigen::Matrix3d& D,
    bool disable_output,
    bool disable_debug_output
);
