#pragma once

#include <Eigen/Dense>

class Crack{
    std::vector<Eigen::Vector2d> points;
    std::vector<Eigen::Vector2i> segments;

    int addPoint(const Eigen::Vector2d& p);
    int addSegment(int p0, int p1);
};

