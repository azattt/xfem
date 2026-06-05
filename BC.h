#pragma once

#include <Eigen/Dense>

void applyBC(double w, double h, int wn, int hn, Eigen::VectorXd& P, const std::vector<unsigned int>& node_offset, std::vector<int>& fixedDofs,
std::vector<double>& fixedValues);