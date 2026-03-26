#include "solver.h"

#include <Eigen/LU>      // for PartialPivLU
#include <Eigen/Cholesky> 
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

Eigen::VectorXd solveLinearSystemLU(const Eigen::MatrixXd& K, const Eigen::VectorXd& P) {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(K);
    return lu.solve(P);
}
Eigen::VectorXd solveLinearSystemLLT(const Eigen::MatrixXd& K, const Eigen::VectorXd& P) {
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> LLT(K);
    return LLT.solve(P);
}
Eigen::VectorXd solveLinearSystemLdLT(const Eigen::MatrixXd& K, const Eigen::VectorXd& P) {
    Eigen::LDLT<Eigen::MatrixXd> LDLT(K);
    return LDLT.solve(P);
}
Eigen::VectorXd solveSparseLinearSystemLLT(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P) {
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> LLT(K);
    return LLT.solve(P);
}
Eigen::VectorXd solveSparseLinearSystemLDLT(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT(K);
    return LDLT.solve(P);
}
Eigen::VectorXd solveSparseLinearSystemLU(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> LU(K);
    return LU.solve(P);
}