#pragma once
#include <Eigen/Dense>  // or just <Eigen/Core> + forward declare
#include <Eigen/Sparse>
Eigen::VectorXd solveLinearSystemLU(const Eigen::MatrixXd& K, const Eigen::VectorXd& P);
Eigen::VectorXd solveLinearSystemLLT(const Eigen::MatrixXd& K, const Eigen::VectorXd& P);
Eigen::VectorXd solveLinearSystemLdLT(const Eigen::MatrixXd& K, const Eigen::VectorXd& P);
Eigen::VectorXd solveSparseLinearSystemLLT(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P);
Eigen::VectorXd solveSparseLinearSystemLDLT(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P);
Eigen::VectorXd solveSparseLinearSystemLU(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& P);