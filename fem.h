#pragma once

#include <array>
#include <variant>
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

enum EnrichmentType
{
    NoEnrichment = 0,
    Heaviside = 1,
    Tip = 2,
};

struct Triangle1
{
    glm::dvec2 v0, v1, v2;
};

struct Triangle1HeavisideSign
{
    glm::dvec2 v0, v1, v2;
    int sign;
};

struct HeavisideEnrichedQuad{
    // // global coords of linear quad in standard order (counter-clockwise)
    // // left-bottom right-bottom right-top left-top
    std::array<unsigned int, 4> node_ids;
    // // local xi, eta of crack with element intersection points (before isoparametric map)
    std::array<glm::vec2, 2> intersection_points_local;
    // // convex non-degenerate quad triangulation consists of four triangles
    // // we store triangles in order where the first $positive_heaviside_triangles_num triangles are above crack
    // // and so a heaviside sign is positive
    // // and the remaining triangles have negative signs
    // // int is used because of performance. if we even use 1 byte, padding will consume remaining 3
    unsigned int positive_heaviside_triangles_num;
    // // this array stores indices
    std::array<std::array<unsigned char, 3>, 4> triangles;
};



struct EnrichedQuad
{
    glm::vec2 v00, v10, v11, v01;
    EnrichmentType enrichment_type;
    std::variant<std::vector<Triangle1HeavisideSign>, std::vector<Triangle1>> triangulation;
};

struct Triangle1HeavisideSignEigen
{
    Eigen::Matrix<double, 3, 2> coordMat;
    int sign;
};

struct Node
{
    double x, y;
};

Eigen::Matrix3d setup_D_matrix(double E, double nu, bool plane_stress);

namespace GeneralFEM{
    struct JacobianData
    {
        Eigen::Matrix2d J;    // Jacobian matrix
        double detJ;          // determinant
        Eigen::Matrix2d invJ; // inverse
    };
}
namespace LinearQuad
{
    using Element = std::array<int, 4>;
    using JacobianData = GeneralFEM::JacobianData;
    // Gauss quadrature data for 2x2 rule
    constexpr int NGauss = 4;                                                         // 2x2 = 4 points
    constexpr std::array<double, 2> gauss_pts{-0.577350269189626, 0.577350269189626}; // ±1/√3
    constexpr std::array<double, 2> gauss_wts{1.0, 1.0}; // weights for 1D, product gives 2D weight

    struct ShapeData
    {
        Eigen::Vector4d N;
        Eigen::Matrix<double, 2, 4> dN_xi_eta; // first row for xi, second row for eta
    };
    std::array<ShapeData, NGauss> precompute_shape_data();
    JacobianData compute_jacobian(const ShapeData &shape, const Eigen::Matrix<double, 4, 2> &coords);
    Eigen::Matrix<double, 3, 8> compute_B_matrix(const ShapeData &shape, const JacobianData &jd);
    Eigen::Matrix<double, 8, 8> element_stiffness(const Eigen::Matrix<double, 4, 2> &coordMat, const Eigen::Matrix3d &D, double h);
} // namespace LinearQuad


namespace LinearTriangle
{
    constexpr unsigned int nNodes = 3;
    constexpr unsigned int nDOFperNode = 2;
    constexpr unsigned int nStrainTensorComponents = 3;
    using JacobianData = GeneralFEM::JacobianData;
// Зенкевич 1975 стр. 82-83
namespace Triangle3PointRule
{
    constexpr unsigned int NGauss = 3;
    constexpr std::array<std::array<double, 2>, NGauss> gauss_pts{{
        {1.0 / 2.0, 1.0 / 2.0},
        {0.0, 1.0 / 2.0},
        {1.0 / 2.0, 0.0},
    }};
    // В Зенкевиче умножены на 2
    constexpr std::array<double, NGauss> gauss_wts{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};
    struct ShapeData
    {
        Eigen::Vector<double, NGauss> N;
        Eigen::Matrix<double, 2, NGauss> dN_xi_eta; // first row for xi, second row for eta
    };
    std::array<ShapeData, NGauss> precompute_shape_data();

    JacobianData compute_jacobian(const ShapeData &shape, const Eigen::Matrix<double, 3, 2> &coords);
    Eigen::Matrix<double, nStrainTensorComponents, nNodes*nDOFperNode> compute_B_matrix(const ShapeData &shape, const JacobianData &jd, int heaviside_func_value);
    Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> element_stiffness(const Eigen::Matrix<double, nNodes, nDOFperNode> &coordMat, const Eigen::Matrix3d &D, double h);
} // namespace TriangleSecondRule

// TODO

namespace Triangle7PointRule
{
    constexpr int NGauss = 7;
    constexpr double a1 = 0.05971587, b1 = 0.47014206, a2 = 0.79742699, b2 = 0.10128651;
    constexpr std::array<std::array<double, 2>, 7> gauss_pts{
        {{1.0 / 3.0, 1.0 / 3.0}, {a1, b1}, {b1, a1}, {b1, b1}, {a2, b2}, {b2, a2}, {b2, b2}}};
    // В Зенкевиче умножены на 2
    constexpr std::array<double, 7> gauss_wts{0.1125, 0.066197075, 0.066197075, 0.066197075, 0.06296959, 0.06296959, 0.06296959};
} // namespace Triangle7PointRule

namespace Triangle13PointRule
{
    constexpr int NGauss = 13;

    // Reference triangle:
    // (r, s), r >= 0, s >= 0, r + s <= 1
    //
    // Weights are scaled so that:
    //
    // sum(gauss_wts) = 0.5
    //
    // Therefore use:
    //
    // factor = w * abs(det_tri) * abs(detJ_quad)

    static const std::array<std::array<double, 2>, NGauss> gauss_pts = {{
        // centroid
        {1.0 / 3.0, 1.0 / 3.0},

        // group 1: a = 0.479308067841920, b = 0.260345966079040
        {0.260345966079040, 0.260345966079040},
        {0.260345966079040, 0.479308067841920},
        {0.479308067841920, 0.260345966079040},

        // group 2: a = 0.869739794195568, b = 0.065130102902216
        {0.065130102902216, 0.065130102902216},
        {0.065130102902216, 0.869739794195568},
        {0.869739794195568, 0.065130102902216},

        // group 3: a = 0.638444188569809,
        //          b = 0.312865496004875,
        //          c = 0.048690315425316
        {0.312865496004875, 0.048690315425316},
        {0.048690315425316, 0.312865496004875},
        {0.638444188569809, 0.048690315425316},
        {0.048690315425316, 0.638444188569809},
        {0.638444188569809, 0.312865496004875},
        {0.312865496004875, 0.638444188569809}
    }};

    static const std::array<double, NGauss> gauss_wts = {{
        // centroid
        -0.0747850222338350,

        // group 1, original weight 0.175615257433208 / 2
        0.0878076287166040,
        0.0878076287166040,
        0.0878076287166040,

        // group 2, original weight 0.053347235608839 / 2
        0.0266736178044195,
        0.0266736178044195,
        0.0266736178044195,

        // group 3, original weight 0.077113760890257 / 2
        0.0385568804451285,
        0.0385568804451285,
        0.0385568804451285,
        0.0385568804451285,
        0.0385568804451285,
        0.0385568804451285
    }};
}

} // namespace LinearTriangle

namespace HeavisideLinearQuad{
    constexpr unsigned int nNodes = 4;
    constexpr unsigned int nDOFperNode = 4;
    constexpr unsigned int nStrainTensorComponents = 3;
    
    using GeneralFEM::JacobianData;

    using LinearTriangle::Triangle3PointRule::NGauss;
    using LinearTriangle::Triangle3PointRule::gauss_wts;
    using LinearTriangle::Triangle3PointRule::gauss_pts;
    using LinearTriangle::Triangle3PointRule::ShapeData;
    using LinearTriangle::Triangle3PointRule::compute_jacobian;

    Eigen::Matrix<double, nStrainTensorComponents, LinearTriangle::nNodes*nDOFperNode> compute_subtriangle_B_matrix(const ShapeData &shape, const JacobianData &jd, int heaviside_func_value);
    Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> element_stiffness(const Eigen::Matrix<double, nNodes, nDOFperNode> &coordMat, const Eigen::Matrix3d &D, double h, std::vector<Triangle1HeavisideSignEigen> triangulation);
};

namespace FEMAssemble
{
    constexpr int UX = 0b1, UY = 0b10;
    constexpr int nDOFperNode = 2;
    void fixDOF(Eigen::MatrixXd &K, Eigen::VectorXd &P, unsigned int node, const int DOFmask);
    constexpr unsigned int dofPerElement = 2 * 4;
    std::vector<Eigen::Triplet<double>> createTripletsUpperStiffness(unsigned int nElements, unsigned int nHeavisideEnriched);
    void addElementSparseUpperStiffness(const LinearQuad::Element& element,
                                    const Eigen::MatrixXd &Ke,
                                    std::vector<Eigen::Triplet<double>>& triplets,
                                    const std::vector<unsigned int>& node_offset,
                                    const std::vector<unsigned int>& node_ndof,
                                    int max_ndof,
                                    std::vector<bool>& active);
    Eigen::SparseMatrix<double> createStiffnessFromTriplets(
        const std::vector<Eigen::Triplet<double>> &triplets, unsigned int nDOFs);
    void applyDirichletSymmetric(Eigen::SparseMatrix<double> &K, Eigen::VectorXd &F, const std::vector<int> &fixedDofs,
                                const std::vector<double> &fixedValues);
    Eigen::VectorXd solveSparseSPDUpper(Eigen::SparseMatrix<double> K, Eigen::VectorXd P);
} // namespace FEMAssemble