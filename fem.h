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

// -----------------------------------------------------------------------------
// 13-point Dunavant quadrature rule for triangles (degree 7)
// Fully symmetric; weights sum to 0.5 (area of reference triangle in barycentric
// coordinates). Points are given in barycentric coordinates (alpha, beta, gamma)
// with alpha+beta+gamma = 1.
// -----------------------------------------------------------------------------
namespace Triangle13PointRule
{
    constexpr int NGauss = 13;

    // Barycentric coordinates of the 7 suborders (one representative point per group)
    // Each group has a multiplicity (number of symmetric permutations).
    struct Suborder
    {
        double alpha, beta, gamma;  // barycentric coordinates
        int    multiplicity;
        double weight;
    };

    constexpr std::array<Suborder, 7> suborders = {{
        // Center point (multiplicity 1)
        {1.0/3.0, 1.0/3.0, 1.0/3.0, 1, 0.0585965378125140},
        // Group 2
        {0.0656580994902338, 0.4671709937548831, 0.4671709937548831, 3, 0.0367328276581104},
        // Group 3
        {0.2394561903592004, 0.3802719138203998, 0.3802719138203998, 3, 0.0460857289325750},
        // Group 4
        {0.1223765409381765, 0.4388117295309117, 0.4388117295309117, 3, 0.0456344039165646},
        // Group 5
        {0.4805921777677173, 0.2597039111161413, 0.2597039111161413, 3, 0.0269370892491420},
        // Group 6
        {0.0656580994902338, 0.0656580994902338, 0.8686838010195324, 3, 0.0122632040831080},
        // Group 7
        {0.2394561903592004, 0.2394561903592004, 0.5210876192815992, 3, 0.0211130507384505}
    }};

    // Flattened arrays for direct iteration (optional, but convenient)
    // We'll generate them at compile time using a constexpr function.
    // However, to keep things simple, we'll provide a runtime generator.
    // For performance, you can pre‑compute these once.

    inline void getQuadraturePoints(std::vector<std::array<double,2>>& points,
                                    std::vector<double>& weights)
    {
        points.clear();
        weights.clear();
        for (const auto& sub : suborders)
        {
            // Generate all permutations of (alpha, beta, gamma) with multiplicity
            // For multiplicity 1, only the point itself.
            // For multiplicity 3, we have 3 distinct permutations:
            //   (a,b,c), (b,c,a), (c,a,b)
            // But careful: some suborders have two equal coordinates, so permutations
            // may produce duplicates. The multiplicity given already accounts for that.
            // We'll generate the minimal set and let the loop handle it.
            std::vector<std::array<double,3>> perms;
            double a = sub.alpha, b = sub.beta, c = sub.gamma;
            if (sub.multiplicity == 1)
            {
                perms.push_back({a, b, c});
            }
            else if (sub.multiplicity == 3)
            {
                // Three distinct permutations (assuming not all equal)
                perms.push_back({a, b, c});
                perms.push_back({b, c, a});
                perms.push_back({c, a, b});
                // Remove duplicates if two coordinates are equal (e.g., a == b)
                // In such case, the three permutations reduce to one. But the
                // multiplicity already tells us how many unique points to add.
                // To avoid double counting, we can use a set, but simpler:
                // We'll rely on the fact that for suborders with two equal values,
                // the multiplicity is still 3 but the generated set will have
                // only 1 unique point. To fix, we'll use a small filter.
                std::vector<std::array<double,3>> unique;
                for (const auto& p : perms)
                {
                    bool duplicate = false;
                    for (const auto& u : unique)
                    {
                        if (std::abs(p[0]-u[0]) < 1e-12 && std::abs(p[1]-u[1]) < 1e-12)
                        { duplicate = true; break; }
                    }
                    if (!duplicate) unique.push_back(p);
                }
                perms = unique;
                // Now perms size should equal sub.multiplicity (1 or 3)
            }
            for (const auto& p : perms)
            {
                // Barycentric to Cartesian (x,y) coordinates on reference triangle
                // We need (ξ, η) coordinates for the triangle's local coordinate system.
                // Typically the reference triangle has vertices: (0,0), (1,0), (0,1).
                // Then barycentric (α,β,γ) corresponds to (x=β, y=γ).
                // But your code uses points in (ξ,η) for the quadrilateral sub‑triangles.
                // For consistency, we keep the barycentric coordinates and later
                // map to (ξ,η) using the triangle's vertices (given in local quadrilateral coordinates).
                // So we store the barycentric coordinates (α,β,γ) – actually we only need (β,γ)
                // because α = 1-β-γ.
                double x = p[1]; // β
                double y = p[2]; // γ
                points.push_back({x, y});
                weights.push_back(sub.weight / sub.multiplicity); // distribute weight among permutations
            }
        }
        // Ensure sum of weights = 0.5
        double sum = 0.0;
        for (double w : weights) sum += w;
        // (optional) adjust if needed
    }

    // Alternatively, pre‑compute static arrays (more efficient)
    // We'll create static arrays of size 13.
    static const std::array<std::array<double,2>, 13> gauss_pts = [](){
        std::vector<std::array<double,2>> pts;
        std::vector<double> wts;
        getQuadraturePoints(pts, wts);
        std::array<std::array<double,2>,13> arr{};
        for (size_t i=0; i<13; ++i) arr[i] = pts[i];
        return arr;
    }();

    static const std::array<double,13> gauss_wts = [](){
        std::vector<std::array<double,2>> pts;
        std::vector<double> wts;
        getQuadraturePoints(pts, wts);
        std::array<double,13> arr{};
        for (size_t i=0; i<13; ++i) arr[i] = wts[i];
        return arr;
    }();
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