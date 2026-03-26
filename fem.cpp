#include "fem.h"


#include <Eigen/SparseCholesky>


Eigen::Matrix3d setup_D_matrix(double E, double nu, bool plane_stress)
{
    Eigen::Matrix3d D;
    if (plane_stress)
    {
        double factor = E / (1 - nu * nu);
        D(0, 0) = factor;
        D(0, 1) = factor * nu;
        D(0, 2) = 0.0;
        D(1, 0) = factor * nu;
        D(1, 1) = factor;
        D(1, 2) = 0.0;
        D(2, 0) = 0.0;
        D(2, 1) = 0.0;
        D(2, 2) = factor * (1 - nu) / 2.0;
    }
    else
    {
        // Plane strain
        double factor = E / ((1 + nu) * (1 - 2 * nu));
        D(0, 0) = factor * (1 - nu);
        D(0, 1) = factor * nu;
        D(0, 2) = 0.0;
        D(1, 0) = factor * nu;
        D(1, 1) = factor * (1 - nu);
        D(1, 2) = 0.0;
        D(2, 0) = 0.0;
        D(2, 1) = 0.0;
        D(2, 2) = factor * (1 - 2 * nu) / 2.0;
    }
    return D;
}

namespace LinearQuad
{
    std::array<ShapeData, NGauss> precompute_shape_data()
{
    std::array<ShapeData, NGauss> data;
    int gp = 0;
    for (int i = 0; i < 2; ++i)
    {
        double xi = gauss_pts[i];
        for (int j = 0; j < 2; ++j)
        {
            double eta = gauss_pts[j];
            ShapeData &d = data[gp++];
            // Node 1
            d.N[0] = 0.25 * (1 - xi) * (1 - eta);
            d.dN_xi_eta(0, 0) = -0.25 * (1 - eta);
            d.dN_xi_eta(1, 0) = -0.25 * (1 - xi);
            // Node 2
            d.N[1] = 0.25 * (1 + xi) * (1 - eta);
            d.dN_xi_eta(0, 1) = 0.25 * (1 - eta);
            d.dN_xi_eta(1, 1) = -0.25 * (1 + xi);
            // Node 3
            d.N[2] = 0.25 * (1 + xi) * (1 + eta);
            d.dN_xi_eta(0, 2) = 0.25 * (1 + eta);
            d.dN_xi_eta(1, 2) = 0.25 * (1 + xi);
            // Node 4
            d.N[3] = 0.25 * (1 - xi) * (1 + eta);
            d.dN_xi_eta(0, 3) = -0.25 * (1 + eta);
            d.dN_xi_eta(1, 3) = 0.25 * (1 - xi);
        }
    }
    return data;
}
    const std::array<ShapeData, NGauss> shape_data = precompute_shape_data(); // pre‑computed once

    JacobianData compute_jacobian(const ShapeData &shape, const Eigen::Matrix<double, 4, 2> &coords)
{
    JacobianData jd;
    // Initialize to zero
    jd.J = shape.dN_xi_eta * coords;
    bool invertible;
    jd.J.computeInverseAndDetWithCheck(jd.invJ, jd.detJ, invertible, 1e-12);
    if (!invertible)
        throw std::runtime_error("Jacobi matrix is not invertible");
    return jd;
}

// TODO: can be unrolled to direct stiffness matrix computation
    Eigen::Matrix<double, 3, 8> compute_B_matrix(const ShapeData &shape, const JacobianData &jd)
    {
        // For each node, compute global derivatives
        Eigen::Matrix<double, 3, 8> B;
        Eigen::Matrix<double, 2, 4> dN_dx_dy;
        dN_dx_dy = jd.invJ * shape.dN_xi_eta;

        // Fill B matrix (3 rows, 8 columns)
        // Row 1 (epsilon_x) : columns for u displacements
        // Row 2 (epsilon_y) : columns for v displacements
        // Row 3 (gamma_xy)  : mixed
        for (int i = 0; i < 4; ++i)
        {
            B(0, 2 * i) = dN_dx_dy(0, i); // du/dx
            B(0, 2 * i + 1) = 0;
            B(1, 2 * i) = 0;
            B(1, 2 * i + 1) = dN_dx_dy(1, i); // dv/dy
            B(2, 2 * i) = dN_dx_dy(1, i);     // dv/dx
            B(2, 2 * i + 1) = dN_dx_dy(0, i); // du/dy
        }
        return B;
    }

    Eigen::Matrix<double, 8, 8> element_stiffness(const Eigen::Matrix<double, 4, 2> &coordMat, const Eigen::Matrix3d &D, double h)
    {
        Eigen::Matrix<double, 8, 8> Ke;
        Ke.setZero();
        // Loop over Gauss points

        int gp = 0;

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                const ShapeData &shape = shape_data[gp++];
                double weight = gauss_wts[i] * gauss_wts[j]; // 2D weight = 1.0 * 1.0 = 1.
                // Compute Jacobian and its determinant
                JacobianData jd = compute_jacobian(shape, coordMat);
                Eigen::Matrix<double, 3, 8> B = compute_B_matrix(shape, jd);
                Ke += B.transpose() * D * B * weight * jd.detJ;
            }
        }
        return Ke * h;
    }
} // namespace LinearQuad

namespace LinearTriangle
{
    namespace TriangleSecondRule{
        std::array<ShapeData, NGauss> precompute_shape_data(){
        std::array<ShapeData, NGauss> data;
        for (int gp = 0; gp < NGauss; gp++){
            double L1 = gauss_pts[gp][0], L2 = gauss_pts[gp][1];
            ShapeData &d = data[gp];
            // Node 1
            d.N[0] = L1;
            d.dN_xi_eta(0, 0) = 1;
            d.dN_xi_eta(1, 0) = 0;
            // Node 2
            d.N[1] = L2;
            d.dN_xi_eta(0, 1) = 0;
            d.dN_xi_eta(1, 1) = 1;
            // Node 3
            d.N[2] = 1 - L1 - L2;
            d.dN_xi_eta(0, 2) = -1;
            d.dN_xi_eta(1, 2) = -1;
        }
        return data;
    }
    const std::array<ShapeData, NGauss> shape_data = precompute_shape_data(); // pre‑computed once
    JacobianData compute_jacobian(const ShapeData &shape, const Eigen::Matrix<double, nNodes, nDOFperNode> &coords)
    {
        JacobianData jd;
        // Initialize to zero
        jd.J = shape.dN_xi_eta * coords;
        bool invertible;
        jd.J.computeInverseAndDetWithCheck(jd.invJ, jd.detJ, invertible, 1e-12);
        if (!invertible)
        throw std::runtime_error("Jacobi matrix is not invertible");
        return jd;
    }
    Eigen::Matrix<double, nStrainTensorComponents, nNodes*nDOFperNode> compute_B_matrix(const ShapeData &shape, const JacobianData &jd)
    {
        // For each node, compute global derivatives
        Eigen::Matrix<double, nStrainTensorComponents, nNodes*nDOFperNode> B; // epsilon_xx epsilon_yy epsilon_xy
        Eigen::Matrix<double, 2, nNodes> dN_dx_dy; // we assume that three node => three shape functions
        dN_dx_dy = jd.invJ * shape.dN_xi_eta;

        // Fill B matrix (3 rows, 8 columns)
        // Row 1 (epsilon_x) : columns for u displacements
        // Row 2 (epsilon_y) : columns for v displacements
        // Row 3 (gamma_xy)  : mixed
        for (int i = 0; i < nNodes; ++i)
        {
            B(0, nDOFperNode * i) = dN_dx_dy(0, i); // du/dx
            B(0, nDOFperNode * i + 1) = 0;
            B(1, nDOFperNode * i) = 0;
            B(1, nDOFperNode * i + 1) = dN_dx_dy(1, i); // dv/dy
            B(2, nDOFperNode * i) = dN_dx_dy(1, i);     // dv/dx
            B(2, nDOFperNode * i + 1) = dN_dx_dy(0, i); // du/dy
        }
        return B;
    }
    Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> element_stiffness(const Eigen::Matrix<double, nNodes, nDOFperNode> &coordMat, const Eigen::Matrix3d &D, double h)
    {
        Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> Ke;
        Ke.setZero();
        // Loop over Gauss points

        for (unsigned int gp = 0; gp < NGauss; gp++){
            const ShapeData &shape = shape_data[gp];
            double weight = gauss_wts[gp];
            // Compute Jacobian and its determinant
            JacobianData jd = compute_jacobian(shape, coordMat);
            Eigen::Matrix<double, nStrainTensorComponents, nNodes*nDOFperNode> B = compute_B_matrix(shape, jd);
            Ke += B.transpose() * D * B * weight * jd.detJ;
        }
        
        return Ke * h;
    }
}
}

namespace HeavisideLinearQuad{
    const std::array<ShapeData, NGauss> shape_data = LinearTriangle::TriangleSecondRule::precompute_shape_data(); // pre‑computed once
    Eigen::Matrix<double, nStrainTensorComponents, LinearTriangle::nNodes*nDOFperNode> compute_subtriangle_B_matrix(const LinearTriangle::TriangleSecondRule::ShapeData &shape, const LinearTriangle::TriangleSecondRule::JacobianData &jd, int heaviside_func_value)
    {
        // For each node, compute global derivatives
        Eigen::Matrix<double, nStrainTensorComponents, LinearTriangle::nNodes*nDOFperNode> B; // epsilon_xx epsilon_yy epsilon_xy
        Eigen::Matrix<double, 2, LinearTriangle::nNodes> dN_dx_dy; // we assume that three node => three shape functions
        dN_dx_dy = jd.invJ * shape.dN_xi_eta;

        // Fill B matrix (3 rows, 8 columns)
        // Row 1 (epsilon_x) : columns for u displacements
        // Row 2 (epsilon_y) : columns for v displacements
        // Row 3 (gamma_xy)  : mixed
        for (unsigned int i = 0; i < LinearTriangle::nNodes; ++i)
        {
            B(0, nDOFperNode * i) = dN_dx_dy(0, i); // du/dx
            B(0, nDOFperNode * i + 1) = 0;
            B(1, nDOFperNode * i) = 0;
            B(1, nDOFperNode * i + 1) = dN_dx_dy(1, i); // dv/dy
            B(2, nDOFperNode * i) = dN_dx_dy(1, i);     // dv/dx
            B(2, nDOFperNode * i + 1) = dN_dx_dy(0, i); // du/dy

            // enriched part
            B(0, nDOFperNode * i + 2) = dN_dx_dy(0, i) * heaviside_func_value; // du/dx
            B(0, nDOFperNode * i + 3) = 0;
            B(1, nDOFperNode * i + 2) = 0;
            B(1, nDOFperNode * i + 3) = dN_dx_dy(1, i) * heaviside_func_value; // dv/dy
            B(2, nDOFperNode * i + 2) = dN_dx_dy(1, i) * heaviside_func_value;     // dv/dx
            B(2, nDOFperNode * i + 3) = dN_dx_dy(0, i) * heaviside_func_value; // du/dy
        }
        return B;
    }
    Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> element_stiffness(const Eigen::Matrix<double, nNodes, 2> &coordMat, const Eigen::Matrix3d &D, double h, std::vector<Triangle1HeavisideSignEigen> triangulation){
        Eigen::Matrix<double, nNodes*nDOFperNode, nNodes*nDOFperNode> Ke;
        Ke.setZero();
        // for (const Triangle1HeavisideSignEigen& triangleHeaviside: triangulation){
            
        //     for (unsigned int gp = 0; gp < NGauss; gp++){
        //         const ShapeData &shape = shape_data[gp];
        //         double weight = gauss_wts[gp];
        //         // Compute Jacobian and its determinant
        //         JacobianData jd = compute_jacobian(shape, coordMat);
        //         Eigen::Matrix<double, nStrainTensorComponents, LinearTriangle::nNodes*nDOFperNode> B = compute_subtriangle_B_matrix(shape, jd, triangleHeaviside.sign);
        //         Ke  += B.transpose() * D * B * weight * jd.detJ;
        //     }
        // }
        // LinearTriangle::TriangleSecondRule
        return Ke;
    }
};
// Eigen::Matrix<double, 3, 16> compute_B_heaviside_matrix(const ShapeData4& shape, const JacobianData& jd) {
//     // For each node, compute global derivatives
//     Eigen::Matrix<double, 3, 16> B;
//     Eigen::Matrix<double, 2, 4> dN_dx_dy;
//     dN_dx_dy = jd.invJ * shape.dN_xi_eta;

//     // Fill B matrix (3 rows, 8 columns)
//     // Row 1 (epsilon_x) : columns for u displacements
//     // Row 2 (epsilon_y) : columns for v displacements
//     // Row 3 (gamma_xy)  : mixed
//     for (int i = 0; i < 4; ++i) {
//         B(0, 4*i)   = dN_dx_dy(0, i); // du/dx
//         B(0, 4*i+1) = 0;
//         B(1, 4*i) = 0;
//         B(1, 4*i+1) = dN_dx_dy(1, i); // dv/dy
//         B(2, 4*i)   = dN_dx_dy(1, i); // dv/dx
//         B(2, 4*i+1) = dN_dx_dy(0, i); // du/dy

//         B(0, 4*i+2)   = dN_dx_dy(0, i); // du/dx
//         B(0, 4*i+3) = 0;
//         B(1, 4*i+2) = 0;
//         B(1, 4*i+3) = dN_dx_dy(1, i); // dv/dy
//         B(2, 4*i+2)   = dN_dx_dy(1, i); // dv/dx
//         B(2, 4*i+3) = dN_dx_dy(0, i); // du/dy
//     }
//     return B;
// }

// Eigen::Matrix<double, 8, 8> element_stiffness_heaviside(const std::vector<Triangle1HeavisideSign>& triangulation,
// const Eigen::Matrix<double, 4, 2>& coordMat, const Eigen::Matrix3d& D) {
//     Eigen::Matrix<double, 8, 8> Ke;
//     Ke.setZero();
//     // Loop over Gauss points

//     for (const Triangle1HeavisideSign& triangle: triangulation){
//         Eigen::Matrix<double, 3, 16> B;
//     }

//     int gp = 0;

//     for (int i = 0; i < 2; ++i) {
//         for (int j = 0; j < 2; ++j) {
//             const ShapeData3& shape = shape_data3[gp++];
//             double weight = gauss_wts[i] * gauss_wts[j];  // 2D weight = 1.0 * 1.0 = 1.
//             // Compute Jacobian and its determinant
//             // JacobianData jd = compute_jacobian(shape, coordMat);
//             // Eigen::Matrix<double, 3, 8> B = compute_B_matrix(shape, jd);
//             // Ke += B.transpose()*D*B*weight * jd.detJ;
//         }
//     }
//     return Ke;
// }
namespace FEMAssemble{

    // for symmetric matrices
    std::vector<Eigen::Triplet<double>> createTripletsUpperStiffness(unsigned int nElements, unsigned int nHeavisideEnriched){
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nElements*dofPerElement*(dofPerElement+1)/2 + nHeavisideEnriched*4*(4+1)/2);
        return triplets;
    } 

void addElementSparseUpperStiffness(const LinearQuad::Element& element,
                                    const Eigen::MatrixXd &Ke,
                                    std::vector<Eigen::Triplet<double>>& triplets,
                                    const std::vector<unsigned int>& node_offset,
                                    const std::vector<unsigned int>& node_ndof,
                                    int max_ndof) {
    constexpr int nNodes = 4;
    for (int i = 0; i < nNodes; ++i) {
        int nodeI = element.node_ids[i];
        int offI = node_offset[nodeI];
        int ndI = node_ndof[nodeI];
        for (int j = 0; j < nNodes; ++j) {
            int nodeJ = element.node_ids[j];
            int offJ = node_offset[nodeJ];
            int ndJ = node_ndof[nodeJ];
            for (int di = 0; di < ndI; ++di) {
                int gi = offI + di;
                for (int dj = 0; dj < ndJ; ++dj) {
                    int gj = offJ + dj;
                    if (gi <= gj) {   // upper triangle only
                        double val = Ke(i * max_ndof + di, j * max_ndof + dj);
                        if (std::abs(val) > 1e-15)
                            triplets.emplace_back(gi, gj, val);
                    }
                }
            }
        }
    }
}
    Eigen::SparseMatrix<double> createStiffnessFromTriplets(const std::vector<Eigen::Triplet<double>>& triplets, unsigned int nDOFs){
        Eigen::SparseMatrix<double> matrix(nDOFs, nDOFs);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        return matrix;
    }

    void applyDirichletSymmetric(Eigen::SparseMatrix<double>& K, Eigen::VectorXd& F,
                                const std::vector<int>& fixedDofs,
                                const std::vector<double>& fixedValues) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(K.nonZeros());
        for (int k = 0; k < K.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }

        // Precompute a map of fixed DOF to prescribed value for quick lookup
        std::unordered_map<int, double> fixedMap;
        for (size_t idx = 0; idx < fixedDofs.size(); ++idx) {
            fixedMap[fixedDofs[idx]] = fixedValues[idx];
        }

        // First pass: adjust RHS using the original off-diagonal entries
        for (const auto& t : triplets) {
            int row = t.row();
            int col = t.col();
            if (row == col) continue; // diagonal handled separately
            auto itRow = fixedMap.find(row);
            auto itCol = fixedMap.find(col);
            if (itRow != fixedMap.end()) {
                // row is fixed -> subtract t.value() * u0 from F(col)
                F(col) -= t.value() * itRow->second;
            }
            if (itCol != fixedMap.end() && row != col) {
                // col is fixed -> subtract t.value() * u0 from F(row)
                // For a symmetric matrix stored only upper, the value t.value()
                // corresponds to K(row,col) for row<col, and K(col,row) is the same.
                F(row) -= t.value() * itCol->second;
            }
        }

        // Second pass: remove all triplets that involve any fixed DOF (except diagonal)
        triplets.erase(std::remove_if(triplets.begin(), triplets.end(),
            [&fixedMap](const Eigen::Triplet<double>& t) {
                return fixedMap.count(t.row()) || fixedMap.count(t.col());
            }), triplets.end());

        // Add diagonal entries for fixed DOFs
        for (const auto& pair : fixedMap) {
            int i = pair.first;
            triplets.emplace_back(i, i, 1.0);
            F(i) = pair.second;   // set RHS to prescribed value
        }

        // Rebuild the matrix from triplets
        K.setFromTriplets(triplets.begin(), triplets.end());
        K.makeCompressed();
    }

    Eigen::VectorXd solveSparseSPDUpper(Eigen::SparseMatrix<double> K, Eigen::VectorXd P){
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> LLT(K);
        return LLT.solve(P);
    }
}
