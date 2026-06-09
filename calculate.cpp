#include "calculate.h"

#include "BC.h"
#include "misc.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

void computeCrackTipDirections(
    const Crack& crack,
    Eigen::Vector2d& crack_tip_1_t,
    Eigen::Vector2d& crack_tip_1_n,
    Eigen::Vector2d& crack_tip_2_t,
    Eigen::Vector2d& crack_tip_2_n
)
{
    if (crack.vertices.size() < 2 || crack.indices.empty())
    {
        throw std::runtime_error("computeCrackTipDirections: crack must contain at least one segment");
    }

    const CrackSegment& first_segment = crack.indices.front();

    const Eigen::Vector2d first_crack_segment =
        crack.vertices[first_segment.v0] -
        crack.vertices[first_segment.v1];

    if (first_crack_segment.norm() < 1e-14)
    {
        throw std::runtime_error("computeCrackTipDirections: zero-length first crack segment");
    }

    crack_tip_1_t = first_crack_segment.normalized();
    crack_tip_1_n = Eigen::Vector2d{
        -crack_tip_1_t.y(),
         crack_tip_1_t.x()
    };

    const CrackSegment& last_segment = crack.indices.back();

    const Eigen::Vector2d last_crack_segment =
        crack.vertices[last_segment.v1] -
        crack.vertices[last_segment.v0];

    if (last_crack_segment.norm() < 1e-14)
    {
        throw std::runtime_error("computeCrackTipDirections: zero-length last crack segment");
    }

    crack_tip_2_t = last_crack_segment.normalized();
    crack_tip_2_n = Eigen::Vector2d{
        -crack_tip_2_t.y(),
         crack_tip_2_t.x()
    };
}

static LinearQuad::ShapeData makeQuadShape(double xi, double eta)
{
    LinearQuad::ShapeData shape;

    shape.N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    shape.N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    shape.N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    shape.N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    shape.dN_xi_eta(0, 0) = -0.25 * (1.0 - eta);
    shape.dN_xi_eta(1, 0) = -0.25 * (1.0 - xi);

    shape.dN_xi_eta(0, 1) = 0.25 * (1.0 - eta);
    shape.dN_xi_eta(1, 1) = -0.25 * (1.0 + xi);

    shape.dN_xi_eta(0, 2) = 0.25 * (1.0 + eta);
    shape.dN_xi_eta(1, 2) = 0.25 * (1.0 + xi);

    shape.dN_xi_eta(0, 3) = -0.25 * (1.0 + eta);
    shape.dN_xi_eta(1, 3) = 0.25 * (1.0 - xi);

    return shape;
}

static double det2(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b
)
{
    return a.x() * b.y() - a.y() * b.x();
}

static Eigen::Matrix<double, 4, 2> elementCoords(
    const QuadMesh& mesh,
    const std::array<int, 4>& element
)
{
    Eigen::Matrix<double, 4, 2> coords;

    for (int i = 0; i < 4; ++i)
    {
        coords.row(i) = mesh.vertices[element[i]];
    }

    return coords;
}

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
)
{
    XFemIterationResult result;

    const int total_vertices =
        static_cast<int>(mesh.vertices.size());

    computeCrackTipDirections(
        crack,
        result.crack_tip_1_t,
        result.crack_tip_1_n,
        result.crack_tip_2_t,
        result.crack_tip_2_n
    );

    result.level_set_fields =
        compute_level_set_fields(mesh, crack);

    result.enriched_elements =
        find_enriched_elements_by_level_set_fields_simple(
            mesh,
            crack,
            result.level_set_fields
        );

    result.enriched_elements_triangulation =
        triangulate_enriched(
            mesh,
            result.enriched_elements,
            result.level_set_fields
        );

    result.node_offset.assign(total_vertices, 0);
    result.node_ndof.assign(total_vertices, 12);

    unsigned int dof_counter = 0;

    for (int n = 0; n < total_vertices; ++n)
    {
        result.node_offset[n] = dof_counter;
        result.node_ndof[n] = 12;
        dof_counter += result.node_ndof[n];
    }

    result.dof_counter = dof_counter;
    result.active.assign(dof_counter, false);

    std::vector<Eigen::Triplet<double>> triplets;

    std::cout
        << "Not enriched "
        << result.enriched_elements.regular.size()
        << " Heaviside: "
        << result.enriched_elements.heaviside_enriched.size()
        << " Tip enriched: "
        << result.enriched_elements.tip_enriched.size()
        << std::endl;

    std::cout
        << "Assembling matrix of size: "
        << dof_counter
        << std::endl;

    // ------------------------------------------------------------
    // Duffy quadrature on [0, 1] x [0, 1].
    // Used for tip elements.
    // ------------------------------------------------------------
    constexpr int DuffyN = 8;

    const std::array<double, DuffyN> duffy_pts = {
        0.0198550717512319,
        0.101666761293187,
        0.237233795041836,
        0.408282678752175,
        0.591717321247825,
        0.762766204958164,
        0.898333238706813,
        0.980144928248768
    };

    const std::array<double, DuffyN> duffy_wts = {
        0.0506142681451881,
        0.111190517226687,
        0.156853322938944,
        0.181341891689181,
        0.181341891689181,
        0.156853322938944,
        0.111190517226687,
        0.0506142681451881
    };

    // ============================================================
    // 1. TIP ELEMENTS
    // ============================================================
    for (int i = 0;
         i < static_cast<int>(result.enriched_elements.tip_enriched.size());
         ++i)
    {
        const TipEnriched& enriched_element =
            result.enriched_elements.tip_enriched[i];

        const std::array<int, 4>& element =
            mesh.elements[enriched_element.id];

        const TipTriangulation& triangulation =
            result.enriched_elements_triangulation
                .tip_enriched_triangulation[i];

        Eigen::Matrix<double, 40, 40> Ke;
        Ke.setZero();

        const Eigen::Matrix<double, 4, 2> coords =
            elementCoords(mesh, element);

        std::array<Eigen::Vector2d, 6> local_points = {
            Eigen::Vector2d{-1.0, -1.0},
            Eigen::Vector2d{ 1.0, -1.0},
            Eigen::Vector2d{ 1.0,  1.0},
            Eigen::Vector2d{-1.0,  1.0},
            enriched_element.intersection_point_local_coords,
            enriched_element.tip_point_local_coords
        };

        const Eigen::Vector2d t_vec =
            (
                enriched_element.tip_index == 1
                    ? result.crack_tip_1_t
                    : result.crack_tip_2_t
            ).normalized();

        const Eigen::Vector2d n_vec =
            (
                enriched_element.tip_index == 1
                    ? result.crack_tip_1_n
                    : result.crack_tip_2_n
            ).normalized();

        const double xi_tip =
            enriched_element.tip_point_local_coords.x();

        const double eta_tip =
            enriched_element.tip_point_local_coords.y();

        const LinearQuad::ShapeData tip_shape =
            makeQuadShape(xi_tip, eta_tip);

        Eigen::Vector2d tip_point_global_coords =
            Eigen::Vector2d::Zero();

        for (int k = 0; k < 4; ++k)
        {
            tip_point_global_coords +=
                tip_shape.N[k] * coords.row(k).transpose();
        }

        std::array<std::array<double, 4>, 4> f_nodes;

        for (int n = 0; n < 4; ++n)
        {
            const Eigen::Vector2d d =
                coords.row(n).transpose() -
                tip_point_global_coords;

            const double x1 = d.dot(t_vec);
            const double x2 = d.dot(n_vec);
            const double radius = std::sqrt(x1 * x1 + x2 * x2);

            if (radius < 1e-30)
            {
                f_nodes[n] = {0.0, 0.0, 0.0, 0.0};
                continue;
            }

            const double theta = std::atan2(x2, x1);
            const double sqrt_r = std::sqrt(radius);
            const double sinhalftheta = std::sin(theta / 2.0);
            const double coshalftheta = std::cos(theta / 2.0);
            const double sintheta = std::sin(theta);

            f_nodes[n] = {
                sqrt_r * sinhalftheta,
                sqrt_r * coshalftheta,
                sqrt_r * sintheta * sinhalftheta,
                sqrt_r * sintheta * coshalftheta
            };
        }

        std::array<double, 4> dfdr;
        std::array<double, 4> dfdtheta;
        std::array<Eigen::Vector2d, 4> df_dx;

        Eigen::Matrix<double, 3, 40> B;

        [[maybe_unused]] double area_duffy = 0.0;

        for (unsigned int j = 0; j < 5; ++j)
        {
            const std::array<unsigned char, 3>& triangle =
                triangulation.tri_indices[j];

            int tip_pos = -1;

            for (int q = 0; q < 3; ++q)
            {
                if (triangle[q] == 5)
                {
                    tip_pos = q;
                    break;
                }
            }

            if (tip_pos == -1)
            {
                throw std::runtime_error(
                    "Duffy transform failed: tip point is not a vertex of tip subtriangle"
                );
            }

            const int idx_A = triangle[tip_pos];
            const int idx_B = triangle[(tip_pos + 1) % 3];
            const int idx_C = triangle[(tip_pos + 2) % 3];

            const Eigen::Vector2d A =
                local_points[idx_A];

            const Eigen::Vector2d Bp =
                local_points[idx_B];

            const Eigen::Vector2d Cp =
                local_points[idx_C];

            const Eigen::Vector2d AB = Bp - A;
            const Eigen::Vector2d AC = Cp - A;

            const double det_duffy_triangle =
                std::abs(det2(AB, AC));

            if (det_duffy_triangle < 1e-14)
            {
                throw std::runtime_error(
                    "Degenerate Duffy subtriangle"
                );
            }

            for (int ir = 0; ir < DuffyN; ++ir)
            {
                const double rho = duffy_pts[ir];
                const double wrho = duffy_wts[ir];

                for (int it = 0; it < DuffyN; ++it)
                {
                    B.setZero();

                    const double tau = duffy_pts[it];
                    const double wtau = duffy_wts[it];

                    const Eigen::Vector2d xi_eta =
                        A + rho * ((1.0 - tau) * AB + tau * AC);

                    const double xi = xi_eta.x();
                    const double eta = xi_eta.y();

                    const LinearQuad::ShapeData shape =
                        makeQuadShape(xi, eta);

                    LinearTriangle::JacobianData jd;
                    jd.J = shape.dN_xi_eta * coords;

                    bool invertible = false;

                    jd.J.computeInverseAndDetWithCheck(
                        jd.invJ,
                        jd.detJ,
                        invertible,
                        1e-12
                    );

                    if (!invertible)
                    {
                        throw std::runtime_error(
                            "Jacobi matrix is not invertible"
                        );
                    }

                    const Eigen::Matrix<double, 2, 4> dN_dx_dy =
                        jd.invJ * shape.dN_xi_eta;

                    Eigen::Vector2d gauss_point_global_coords =
                        Eigen::Vector2d::Zero();

                    for (int k = 0; k < 4; ++k)
                    {
                        gauss_point_global_coords +=
                            shape.N[k] * coords.row(k).transpose();
                    }

                    const Eigen::Vector2d d =
                        gauss_point_global_coords -
                        tip_point_global_coords;

                    const double x1 = d.dot(t_vec);
                    const double x2 = d.dot(n_vec);
                    const double radius2 = x1 * x1 + x2 * x2;
                    const double radius = std::sqrt(radius2);

                    if (radius < 1e-14)
                    {
                        continue;
                    }

                    const double theta = std::atan2(x2, x1);

                    const double sqrt_r = std::sqrt(radius);
                    const double inv_sqrt_r = 1.0 / sqrt_r;

                    const double sinhalftheta =
                        std::sin(theta / 2.0);

                    const double coshalftheta =
                        std::cos(theta / 2.0);

                    const double sintheta =
                        std::sin(theta);

                    const double costheta =
                        std::cos(theta);

                    std::array<double, 4> f = {
                        sqrt_r * sinhalftheta,
                        sqrt_r * coshalftheta,
                        sqrt_r * sintheta * sinhalftheta,
                        sqrt_r * sintheta * coshalftheta
                    };

                    dfdr[0] =
                        0.5 * inv_sqrt_r * sinhalftheta;

                    dfdr[1] =
                        0.5 * inv_sqrt_r * coshalftheta;

                    dfdr[2] =
                        0.5 * inv_sqrt_r * sinhalftheta * sintheta;

                    dfdr[3] =
                        0.5 * inv_sqrt_r * coshalftheta * sintheta;

                    dfdtheta[0] =
                        sqrt_r * 0.5 * coshalftheta;

                    dfdtheta[1] =
                        -sqrt_r * 0.5 * sinhalftheta;

                    dfdtheta[2] =
                        sqrt_r *
                        (
                            0.5 * coshalftheta * sintheta +
                            sinhalftheta * costheta
                        );

                    dfdtheta[3] =
                        sqrt_r *
                        (
                            -0.5 * sinhalftheta * sintheta +
                            coshalftheta * costheta
                        );

                    const double drdx = d.x() / radius;
                    const double drdy = d.y() / radius;

                    const double dthetadx =
                        (x1 * n_vec.x() - x2 * t_vec.x()) /
                        radius2;

                    const double dthetady =
                        (x1 * n_vec.y() - x2 * t_vec.y()) /
                        radius2;

                    for (int a = 0; a < 4; ++a)
                    {
                        df_dx[a].x() =
                            dfdr[a] * drdx +
                            dfdtheta[a] * dthetadx;

                        df_dx[a].y() =
                            dfdr[a] * drdy +
                            dfdtheta[a] * dthetady;
                    }

                    for (int n = 0; n < 4; ++n)
                    {
                        const double dNdx = dN_dx_dy(0, n);
                        const double dNdy = dN_dx_dy(1, n);
                        const double Nn = shape.N[n];

                        B(0, 10 * n) = dNdx;
                        B(0, 10 * n + 1) = 0.0;

                        B(1, 10 * n) = 0.0;
                        B(1, 10 * n + 1) = dNdy;

                        B(2, 10 * n) = dNdy;
                        B(2, 10 * n + 1) = dNdx;

                        for (int a = 0; a < 4; ++a)
                        {
                            const double shift =
                                f[a] - f_nodes[n][a];

                            const double d_enr_dx =
                                dNdx * shift +
                                Nn * df_dx[a].x();

                            const double d_enr_dy =
                                dNdy * shift +
                                Nn * df_dx[a].y();

                            B(0, 10 * n + 2 + 2 * a) =
                                d_enr_dx;

                            B(1, 10 * n + 2 + 2 * a) =
                                0.0;

                            B(2, 10 * n + 2 + 2 * a) =
                                d_enr_dy;

                            B(0, 10 * n + 3 + 2 * a) =
                                0.0;

                            B(1, 10 * n + 3 + 2 * a) =
                                d_enr_dy;

                            B(2, 10 * n + 3 + 2 * a) =
                                d_enr_dx;
                        }
                    }

                    const double factor =
                        wrho *
                        wtau *
                        rho *
                        det_duffy_triangle *
                        std::abs(jd.detJ);

                    Ke +=
                        factor *
                        (B.transpose() * D * B) *
                        thickness;

                    area_duffy += factor;
                }
            }
        }

        Eigen::Matrix<double, 48, 48> Ke_expanded;
        Ke_expanded.setZero();

        for (int row_node = 0; row_node < 4; ++row_node)
        {
            for (int col_node = 0; col_node < 4; ++col_node)
            {
                for (int local_row = 0; local_row < 10; ++local_row)
                {
                    for (int local_col = 0; local_col < 10; ++local_col)
                    {
                        const int global_row_dof =
                            local_row < 2
                                ? local_row
                                : local_row + 2;

                        const int global_col_dof =
                            local_col < 2
                                ? local_col
                                : local_col + 2;

                        Ke_expanded(
                            12 * row_node + global_row_dof,
                            12 * col_node + global_col_dof
                        ) =
                            Ke(
                                10 * row_node + local_row,
                                10 * col_node + local_col
                            );
                    }
                }
            }
        }

        FEMAssemble::addElementSparseUpperStiffness(
            LinearQuad::Element{
                element[0],
                element[1],
                element[2],
                element[3]
            },
            Ke_expanded,
            triplets,
            result.node_offset,
            result.node_ndof,
            12,
            result.active
        );
    }

    // ============================================================
    // 2. HEAVISIDE ELEMENTS
    // ============================================================
    for (int i = 0;
         i < static_cast<int>(result.enriched_elements.heaviside_enriched.size());
         ++i)
    {
        const HeavisideEnriched& enriched_element =
            result.enriched_elements.heaviside_enriched[i];

        const std::array<int, 4>& element =
            mesh.elements[enriched_element.id];

        const HeavisideTriangulation& triangulation =
            result.enriched_elements_triangulation
                .heaviside_enriched_triangulation[i];

        Eigen::Matrix<double, 16, 16> Ke;
        Ke.setZero();

        const Eigen::Matrix<double, 4, 2> coords =
            elementCoords(mesh, element);

        std::array<Eigen::Vector2d, 6> local_points = {
            Eigen::Vector2d{-1.0, -1.0},
            Eigen::Vector2d{ 1.0, -1.0},
            Eigen::Vector2d{ 1.0,  1.0},
            Eigen::Vector2d{-1.0,  1.0},
            enriched_element.intersection_points_local_coords[0],
            enriched_element.intersection_points_local_coords[1]
        };

        std::array<int, 4> node_signs;

        for (int n = 0; n < 4; ++n)
        {
            node_signs[n] =
                result.level_set_fields
                    .vertices_level_set_signs[element[n]]
                    .sign;
        }

        [[maybe_unused]] double total_area = 0.0;

        for (unsigned int tri_id = 0;
             tri_id < triangulation.triangles_num;
             ++tri_id)
        {
            const std::array<unsigned char, 3>& triangle =
                triangulation.tri_indices[tri_id];

            int sign = +1;

            if (tri_id >= triangulation.positive_heaviside_triangles_num)
            {
                sign = -1;
            }

            Eigen::Matrix2d J_xieta_rs;
            J_xieta_rs <<
                local_points[triangle[1]].x() -
                    local_points[triangle[0]].x(),
                local_points[triangle[2]].x() -
                    local_points[triangle[0]].x(),
                local_points[triangle[1]].y() -
                    local_points[triangle[0]].y(),
                local_points[triangle[2]].y() -
                    local_points[triangle[0]].y();

            const double det_tri =
                J_xieta_rs.determinant();

            for (unsigned int gp = 0;
                 gp < LinearTriangle::Triangle3PointRule::NGauss;
                 ++gp)
            {
                const double r =
                    LinearTriangle::Triangle3PointRule::gauss_pts[gp][0];

                const double s =
                    LinearTriangle::Triangle3PointRule::gauss_pts[gp][1];

                const double t = 1.0 - r - s;

                const double xi =
                    local_points[triangle[0]].x() * t +
                    local_points[triangle[1]].x() * r +
                    local_points[triangle[2]].x() * s;

                const double eta =
                    local_points[triangle[0]].y() * t +
                    local_points[triangle[1]].y() * r +
                    local_points[triangle[2]].y() * s;

                const LinearQuad::ShapeData shape =
                    makeQuadShape(xi, eta);

                LinearTriangle::JacobianData jd;
                jd.J = shape.dN_xi_eta * coords;

                bool invertible = false;

                jd.J.computeInverseAndDetWithCheck(
                    jd.invJ,
                    jd.detJ,
                    invertible,
                    1e-12
                );

                if (!invertible)
                {
                    throw std::runtime_error(
                        "Jacobi matrix is not invertible"
                    );
                }

                const Eigen::Matrix<double, 2, 4> dN_dx_dy =
                    jd.invJ * shape.dN_xi_eta;

                Eigen::Matrix<double, 3, 16> B;
                B.setZero();

                for (int n = 0; n < 4; ++n)
                {
                    B(0, 4 * n) =
                        dN_dx_dy(0, n);

                    B(1, 4 * n + 1) =
                        dN_dx_dy(1, n);

                    B(2, 4 * n) =
                        dN_dx_dy(1, n);

                    B(2, 4 * n + 1) =
                        dN_dx_dy(0, n);

                    const double shifted_factor =
                        static_cast<double>(sign - node_signs[n]);

                    B(0, 4 * n + 2) =
                        dN_dx_dy(0, n) * shifted_factor;

                    B(1, 4 * n + 3) =
                        dN_dx_dy(1, n) * shifted_factor;

                    B(2, 4 * n + 2) =
                        dN_dx_dy(1, n) * shifted_factor;

                    B(2, 4 * n + 3) =
                        dN_dx_dy(0, n) * shifted_factor;
                }

                const double factor =
                    LinearTriangle::Triangle3PointRule::gauss_wts[gp] *
                    std::abs(det_tri) *
                    std::abs(jd.detJ);

                Ke +=
                    factor *
                    (B.transpose() * D * B) *
                    thickness;

                total_area += det_tri;
            }
        }

        if (!disable_debug_output)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 16, 16>> es(Ke);

            std::cout
                << "eigenvalues of heaviside element: "
                << es.eigenvalues()
                << std::endl;

            std::cout
                << "total_area: "
                << total_area
                << std::endl;

            std::cout
                << "det: "
                << Ke.determinant()
                << std::endl;
        }

        Eigen::Matrix<double, 48, 48> Ke_expanded;
        Ke_expanded.setZero();

        for (int row_node = 0; row_node < 4; ++row_node)
        {
            for (int col_node = 0; col_node < 4; ++col_node)
            {
                for (int local_row = 0; local_row < 4; ++local_row)
                {
                    for (int local_col = 0; local_col < 4; ++local_col)
                    {
                        Ke_expanded(
                            12 * row_node + local_row,
                            12 * col_node + local_col
                        ) =
                            Ke(
                                4 * row_node + local_row,
                                4 * col_node + local_col
                            );
                    }
                }
            }
        }

        FEMAssemble::addElementSparseUpperStiffness(
            LinearQuad::Element{
                element[0],
                element[1],
                element[2],
                element[3]
            },
            Ke_expanded,
            triplets,
            result.node_offset,
            result.node_ndof,
            12,
            result.active
        );
    }

    // ============================================================
    // 3. REGULAR ELEMENTS
    // ============================================================
    Eigen::Matrix<double, 8, 8> Ke_regular;
    Eigen::Matrix<double, 4, 2> coordMat;

    unsigned int elementsCreated = 0;
    const unsigned int elementsTotal =
        static_cast<unsigned int>(mesh.elements.size());

    unsigned int percent = 0;
    unsigned int lastPercent = 0;

    for (const int element_id : result.enriched_elements.regular)
    {
        const std::array<int, 4> element =
            mesh.elements[element_id];

        percent =
            static_cast<unsigned int>(
                100.0 *
                static_cast<double>(elementsCreated) /
                std::max(1u, elementsTotal)
            );

        if (percent > lastPercent)
        {
            std::cout << percent << "% ";
            lastPercent = percent;
        }

        for (int n = 0; n < 4; ++n)
        {
            coordMat(n, 0) =
                mesh.vertices[element[n]].x();

            coordMat(n, 1) =
                mesh.vertices[element[n]].y();
        }

        Ke_regular =
            LinearQuad::element_stiffness(
                coordMat,
                D,
                thickness
            );

        Eigen::Matrix<double, 48, 48> Ke_expanded;
        Ke_expanded.setZero();

        for (int row_node = 0; row_node < 4; ++row_node)
        {
            for (int col_node = 0; col_node < 4; ++col_node)
            {
                for (int local_row = 0; local_row < 2; ++local_row)
                {
                    for (int local_col = 0; local_col < 2; ++local_col)
                    {
                        Ke_expanded(
                            12 * row_node + local_row,
                            12 * col_node + local_col
                        ) =
                            Ke_regular(
                                2 * row_node + local_row,
                                2 * col_node + local_col
                            );
                    }
                }
            }
        }

        FEMAssemble::addElementSparseUpperStiffness(
            LinearQuad::Element{
                element[0],
                element[1],
                element[2],
                element[3]
            },
            Ke_expanded,
            triplets,
            result.node_offset,
            result.node_ndof,
            12,
            result.active
        );

        ++elementsCreated;
    }

    std::cout << std::endl;

    auto K =
        FEMAssemble::createStiffnessFromTriplets(
            triplets,
            dof_counter
        );

    std::cout
        << "Global stiffness matrix is assembled"
        << std::endl;

    if (!disable_output)
    {
        std::cout << "Writing to file." << std::endl;

        Eigen::MatrixXd spK =
            Eigen::MatrixXd(K);

        std::ofstream KoutFile("K.txt");
        KoutFile << spK;
    }

    std::cout << "Creating RHS" << std::endl;

    Eigen::VectorXd P(dof_counter);
    P.setZero();

    std::cout << "Applying boundary conditions" << std::endl;

    std::vector<int> fixedDofs;
    std::vector<double> fixedValues;

    applyBC(
        w,
        h,
        wn,
        hn,
        thickness,
        P,
        result.node_offset,
        fixedDofs,
        fixedValues
    );

    FEMAssemble::applyDirichletSymmetric(
        K,
        P,
        fixedDofs,
        fixedValues
    );

    if (!disable_output)
    {
        std::cout << "Writing to file" << std::endl;

        std::ostringstream KBCout;
        KBCout << Eigen::MatrixXd(K);

        std::ofstream KBCoutFile("KBC.txt");
        KBCoutFile << KBCout.str();

        std::ostringstream Pout;
        Pout << P;

        std::ofstream PoutFile("P.txt");
        PoutFile << Pout.str();
    }

    for (int i = 0; i < static_cast<int>(dof_counter); ++i)
    {
        if (!result.active[i])
        {
            K.coeffRef(i, i) = 1.0;
            P(i) = 0.0;
        }
    }

    std::cout << "Solving linear system" << std::endl;

    result.u =
        FEMAssemble::solveSparseSPDUpper(
            K,
            P
        );

    result.energy =
        0.5 *
        result.u.dot(
            K.selfadjointView<Eigen::Upper>() *
            result.u
        );

    std::cout
        << "Energy: "
        << result.energy
        << std::endl;

    const Eigen::VectorXd residual =
        K.selfadjointView<Eigen::Upper>() *
        result.u -
        P;

    result.residual_norm =
        residual.norm();

    std::cout
        << "||Ku - P|| = "
        << result.residual_norm
        << std::endl;

    return result;
}
