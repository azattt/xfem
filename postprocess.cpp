#include "postprocess.h"

#include "fem.h"
#include "misc.h"

constexpr double PI = 3.14159265358979323846;
constexpr double INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2π)

void drawHeavisideElements(const std::vector<HeavisideEnriched> &heaviside_enriched, const QuadMesh &mesh,
                           const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation,
                           const std::vector<LevelSetSign> &vertices_level_set_signs, const Eigen::VectorXd &u_solu,
                           const std::vector<unsigned int> &node_offset,
                           const std::vector<bool> &heaviside_enriched_nodes, double scale)
{
    for (int i = 0; i < heaviside_enriched.size(); i++)
    {
        const HeavisideEnriched &hvsd_enr = heaviside_enriched[i];
        const std::array<int, 4> &element = mesh.elements[hvsd_enr.id];
        const HeavisideTriangulation &hvsd_trng = heaviside_enriched_triangulation[i];
        std::array<Eigen::Vector2d, 6> points;
        points[0] = Eigen::Vector2d(-1, -1);
        points[1] = Eigen::Vector2d(1, -1);
        points[2] = Eigen::Vector2d(1, 1);
        points[3] = Eigen::Vector2d(-1, 1);
        points[4] = hvsd_enr.intersection_points_local_coords[0];
        points[5] = hvsd_enr.intersection_points_local_coords[1];

        Eigen::Matrix<double, 4, 2> u;
        u.row(0) = u_solu.segment<2>(node_offset[element[0]]) * scale + mesh.vertices[element[0]];
        u.row(1) = u_solu.segment<2>(node_offset[element[1]]) * scale + mesh.vertices[element[1]];
        u.row(2) = u_solu.segment<2>(node_offset[element[2]]) * scale + mesh.vertices[element[2]];
        u.row(3) = u_solu.segment<2>(node_offset[element[3]]) * scale + mesh.vertices[element[3]];
        Eigen::Matrix<double, 4, 2> a;
        for (int n = 0; n < 4; n++)
        {
            if (heaviside_enriched_nodes[element[n]])
            {
                // TODO: fix for variable amount of DOFs
                // a.col(0) = u[node_offset[element[0]]];
                a.row(n) = u_solu.segment<2>(node_offset[element[n]] + 2) * scale;
            }
            else
            {
                a.row(n).setZero();
            }
        }
        Eigen::RowVector4d factor;
        for (int j = 0; j < 4; j++)
        {
            factor(j) = 1 - vertices_level_set_signs[element[j]].sign;
        }
        for (int j = 0; j < hvsd_trng.positive_heaviside_triangles_num; j++)
        {
            std::array<Eigen::Vector2d, 3> v;
            for (int k = 0; k < 3; k++)
            {
                double xi = points[hvsd_trng.tri_indices[j][k]].x(), eta = points[hvsd_trng.tri_indices[j][k]].y();
                Eigen::RowVector4d N{(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta),
                                     (1 - xi) * (1 + eta)};
                N /= 4;
                v[k] = N * u;
                for (int i = 0; i < 4; ++i)
                {
                    v[k] += N[i] * factor[i] * a.row(i);
                }
            }
            TriangleGUI::Renderer::instance().addTriangle(TriangleGUI::TriangleColored{
                toGlm(v[0].cast<float>().eval()), toGlm(v[1].cast<float>().eval()), toGlm(v[2].cast<float>().eval()),
                TriangleGUI::packColor(glm::vec4{0.0, 1.0, 0.0, 1.0})});
        }
        for (int j = 0; j < 4; j++)
        {
            factor(j) = -1 - vertices_level_set_signs[element[j]].sign;
        }
        for (int j = hvsd_trng.positive_heaviside_triangles_num; j < hvsd_trng.triangles_num; j++)
        {
            std::array<Eigen::Vector2d, 3> v;
            for (int k = 0; k < 3; k++)
            {
                double xi = points[hvsd_trng.tri_indices[j][k]].x(), eta = points[hvsd_trng.tri_indices[j][k]].y();
                Eigen::RowVector4d N{(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta),
                                     (1 - xi) * (1 + eta)};
                N /= 4;
                v[k] = N * u;
                for (int i = 0; i < 4; ++i)
                {
                    v[k] += N[i] * factor[i] * a.row(i);
                }
            }
            TriangleGUI::Renderer::instance().addTriangle(TriangleGUI::TriangleColored{
                toGlm(v[0].cast<float>().eval()), toGlm(v[1].cast<float>().eval()), toGlm(v[2].cast<float>().eval()),
                TriangleGUI::packColor(glm::vec4{0.0, 0.0, 1.0, 1.0})});
        }
    }
}

void drawTipElements(const std::vector<TipEnriched> &tip_enriched, const QuadMesh &mesh, const Eigen::VectorXd &u_solu,
                     const std::vector<unsigned int> &node_offset, const std::vector<bool> &heaviside_enriched_nodes,
                     const std::vector<bool> &tip_enriched_nodes, double scale,
                     std::vector<PolygonalChain> &polygonal_chains, const Eigen::Vector2d &crack_tip_1_t,
                     const Eigen::Vector2d &crack_tip_1_n, const Eigen::Vector2d &crack_tip_2_t,
                     const Eigen::Vector2d &crack_tip_2_n)
{
    for (int i = 0; i < tip_enriched.size(); i++)
    {
        const TipEnriched &tip_enr = tip_enriched[i];
        const std::array<int, 4> &element = mesh.elements[tip_enr.id];
        // const TipTriangulation& tip_trng = tip_enriched_triangulation[i];
        constexpr int N_points = 100;
        const Eigen::Vector2d tip_point_global_coords =
            mesh.vertices[element[0]] * (1 - tip_enr.tip_point_local_coords.x()) *
                (1 - tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[1]] * (1 + tip_enr.tip_point_local_coords.x()) *
                (1 - tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[2]] * (1 + tip_enr.tip_point_local_coords.x()) *
                (1 + tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[3]] * (1 - tip_enr.tip_point_local_coords.x()) *
                (1 + tip_enr.tip_point_local_coords.y()) / 4;
        const Eigen::Vector2d intersection_point_global_coords =
            mesh.vertices[element[0]] * (1 - tip_enr.intersection_point_local_coords.x()) *
                (1 - tip_enr.intersection_point_local_coords.y()) / 4 +
            mesh.vertices[element[1]] * (1 + tip_enr.intersection_point_local_coords.x()) *
                (1 - tip_enr.intersection_point_local_coords.y()) / 4 +
            mesh.vertices[element[2]] * (1 + tip_enr.intersection_point_local_coords.x()) *
                (1 + tip_enr.intersection_point_local_coords.y()) / 4 +
            mesh.vertices[element[3]] * (1 - tip_enr.intersection_point_local_coords.x()) *
                (1 + tip_enr.intersection_point_local_coords.y()) / 4;
        const Eigen::Vector2d crack_dir = tip_point_global_coords - intersection_point_global_coords;
        const Eigen::Vector2d crack_dir_local =
            tip_enr.tip_point_local_coords - tip_enr.intersection_point_local_coords;
        const double L = crack_dir_local.norm();
        Eigen::Matrix<double, 4, 2> u;
        u.row(0) = u_solu.segment<2>(node_offset[element[0]]) * scale;
        u.row(1) = u_solu.segment<2>(node_offset[element[1]]) * scale;
        u.row(2) = u_solu.segment<2>(node_offset[element[2]]) * scale;
        u.row(3) = u_solu.segment<2>(node_offset[element[3]]) * scale;
        std::array<Eigen::Matrix<double, 4, 2>, 4> b_nodes; // a_nodes[n][a][component]
        for (int n = 0; n < 4; ++n)
        {
            if (!tip_enriched_nodes[element[n]])
            {
                throw std::runtime_error("NotImplemented: not tip enriched node!!!");
            }
            int base = node_offset[element[n]];
            // int offset = 2;
            // if (heaviside_enriched_nodes[element[n]]) offset += 2;
            int offset = 4;
            for (int a = 0; a < 4; ++a)
            {
                b_nodes[n](a, 0) = u_solu(base + offset + 2 * a) * scale;
                b_nodes[n](a, 1) = u_solu(base + offset + 2 * a + 1) * scale;
            }
        }
        std::vector<glm::vec2> upper_points;
        std::vector<glm::vec2> lower_points;
        Eigen::Matrix<double, 4, 2> vertices;
        vertices.row(0) = mesh.vertices[element[0]];
        vertices.row(1) = mesh.vertices[element[1]];
        vertices.row(2) = mesh.vertices[element[2]];
        vertices.row(3) = mesh.vertices[element[3]];

        std::array<std::array<double, 4>, 4> f_nodes;
        Eigen::Vector2d d;
        double radius, radius2, theta, sqrt_r, sinhalftheta, sintheta, coshalftheta, costheta;
        for (int n = 0; n < 4; n++)
        {
            double xi_tip = tip_enr.tip_point_local_coords.x();
            double eta_tip = tip_enr.tip_point_local_coords.y();
            std::array<double, 4> N_tip;

            d = (vertices.row(n).transpose() - tip_point_global_coords);
            radius = d.norm();

            if (tip_enr.tip_index == 1)
            {
                theta = std::atan2(d.dot(crack_tip_1_n), d.dot(crack_tip_1_t));
            }
            else if (tip_enr.tip_index == 2)
            {
                theta = std::atan2(d.dot(crack_tip_2_n), d.dot(crack_tip_2_t));
            }

            sqrt_r = std::sqrt(radius);
            sinhalftheta = std::sin(theta / 2);
            sintheta = std::sin(theta);
            coshalftheta = std::cos(theta / 2);
            f_nodes[n] = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta, sqrt_r * sintheta * sinhalftheta,
                          sqrt_r * sintheta * coshalftheta};
        }
        for (int i = 0; i < N_points; i++)
        {
            double xi = tip_enr.intersection_point_local_coords.x() + crack_dir_local.x() * i / (N_points);
            double eta = tip_enr.intersection_point_local_coords.y() + crack_dir_local.y() * i / (N_points);
            const Eigen::RowVector4d N{(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, (1 + xi) * (1 + eta) / 4,
                                       (1 - xi) * (1 + eta) / 4};
            const Eigen::RowVector2d p = N * vertices;
            Eigen::Vector2d d_phys = p.transpose() - tip_point_global_coords;
            double r = d_phys.norm();

            const double sqrt_r = std::sqrt(r);
            const double sinhalftheta = 1;
            const double sintheta = 0;
            const double coshalftheta = 0;
            const std::array<double, 4> f = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta,
                                             sqrt_r * sintheta * sinhalftheta, sqrt_r * sintheta * coshalftheta};

            Eigen::RowVector2d p0 = p + N * u;
            for (int n = 0; n < 4; n++)
            {
                for (int a = 0; a < 4; a++)
                {
                    p0 += N[n] * (f[a] - f_nodes[n][a]) * b_nodes[n].row(a);
                }
            }
            upper_points.push_back(glm::vec2{static_cast<float>(p0.x()), static_cast<float>(p0.y())});
        }
        for (int i = N_points; i >= 0; i--)
        {
            double xi = tip_enr.intersection_point_local_coords.x() + crack_dir_local.x() * i / (N_points);
            double eta = tip_enr.intersection_point_local_coords.y() + crack_dir_local.y() * i / (N_points);
            const Eigen::RowVector4d N{(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, (1 + xi) * (1 + eta) / 4,
                                       (1 - xi) * (1 + eta) / 4};
            const Eigen::RowVector2d p = N * vertices;
            Eigen::Vector2d d_phys = p.transpose() - tip_point_global_coords;
            double r = d_phys.norm();
            Eigen::Vector2d to_intersection = intersection_point_global_coords - tip_point_global_coords;
            const double sqrt_r = std::sqrt(r);
            const double sinhalftheta = -1;
            const double sintheta = 0;
            const double coshalftheta = 0;
            const std::array<double, 4> f = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta,
                                             sqrt_r * sintheta * sinhalftheta, sqrt_r * sintheta * coshalftheta};

            Eigen::RowVector2d p0 = p + N * u;
            for (int n = 0; n < 4; n++)
            {
                for (int a = 0; a < 4; a++)
                {
                    p0 += N[n] * (f[a] - f_nodes[n][a]) * b_nodes[n].row(a);
                }
            }
            lower_points.push_back(glm::vec2{static_cast<float>(p0.x()), static_cast<float>(p0.y())});
        }
        PolygonalChain poly_chain_upper;
        poly_chain_upper.color = glm::vec4(1.0, 0.0, 1.0, 1.0);
        poly_chain_upper.points = upper_points;
        polygonal_chains.push_back(poly_chain_upper);
        PolygonalChain poly_chain_lower;
        poly_chain_lower.color = glm::vec4(1.0, 1.0, 0.0, 1.0);
        poly_chain_lower.points = lower_points;
        polygonal_chains.push_back(poly_chain_lower);
    }
}

template <unsigned int NGauss>
void computeStress(const std::vector<TipEnriched> &tip_enriched,
                   const std::vector<HeavisideEnriched> &heaviside_enriched, const QuadMesh &mesh,
                   const std::vector<TipTriangulation> &tip_enriched_triangulation,
                   const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation,
                   const Eigen::VectorXd &u_solu, const std::vector<unsigned int> &node_offset,
                   const std::vector<bool> &heaviside_enriched_nodes, const std::vector<bool> &tip_enriched_nodes,
                   const std::vector<LevelSetSign> &vertices_level_set_signs,
                   const std::array<std::array<double, 2>, NGauss> &gauss_pts,
                   const std::array<double, NGauss> &gauss_wts, const Eigen::Vector2d &crack_tip_1_t,
                   const Eigen::Vector2d &crack_tip_1_n, const Eigen::Vector2d &crack_tip_2_t,
                   const Eigen::Vector2d &crack_tip_2_n, const Eigen::Matrix3d &D, const double young_modulus,
                   const double poisson_ratio, const double Rin, const double Rout)
{
    const double E = young_modulus;
    const double nu = poisson_ratio;
    const double shear_modulus = E / (2.0 * (1.0 + nu));
    const double kappa = (3.0 - nu) / (1.0 + nu); // plane stress

    const double invE = 1.0 / E;

    auto fillQuadShape = [](double xi, double eta) {
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
    };

    auto elementSize = [](const Eigen::Matrix<double, 4, 2> &coords) {
        const double h01 = (coords.row(1) - coords.row(0)).norm();
        const double h12 = (coords.row(2) - coords.row(1)).norm();
        const double h23 = (coords.row(3) - coords.row(2)).norm();
        const double h30 = (coords.row(0) - coords.row(3)).norm();

        return 0.25 * (h01 + h12 + h23 + h30);
    };

    auto elementIntersectsRadius = [](const Eigen::Matrix<double, 4, 2> &coords, const Eigen::Vector2d &tip,
                                      double radius) {
        Eigen::Vector2d center = Eigen::Vector2d::Zero();

        for (int i = 0; i < 4; ++i)
        {
            center += coords.row(i).transpose();
        }

        center /= 4.0;

        double elem_radius = 0.0;

        for (int i = 0; i < 4; ++i)
        {
            elem_radius = std::max(elem_radius, (coords.row(i).transpose() - center).norm());
        }

        return (center - tip).norm() <= radius + elem_radius;
    };
    std::vector<int> heaviside_element_to_index(mesh.elements.size(), -1);

    for (int i = 0; i < static_cast<int>(heaviside_enriched.size()); ++i)
    {
        const int element_id = heaviside_enriched[i].id;

        if (element_id < 0 || element_id >= static_cast<int>(mesh.elements.size()))
        {
            throw std::runtime_error("Invalid Heaviside enriched element id");
        }

        heaviside_element_to_index[element_id] = i;
    }
    for (size_t tip_idx = 0; tip_idx < tip_enriched.size(); ++tip_idx)
    {
        const TipEnriched &tip_data = tip_enriched[tip_idx];
        const std::array<int, 4> &tip_element = mesh.elements[tip_data.id];

        const Eigen::Vector2d t_vec = (tip_data.tip_index == 1 ? crack_tip_1_t : crack_tip_2_t).normalized();

        const Eigen::Vector2d n_vec = (tip_data.tip_index == 1 ? crack_tip_1_n : crack_tip_2_n).normalized();

        Eigen::Matrix2d R;
        R.col(0) = t_vec; // local x1
        R.col(1) = n_vec; // local x2
        // std::cout << "tip_index = " << static_cast<int>(tip_data.tip_index) << "\n";
        // // std::cout << "tip_point = " << tip_point_global.transpose() << "\n";
        // std::cout << "t_vec = " << t_vec.transpose() << "\n";
        // std::cout << "n_vec = " << n_vec.transpose() << "\n";
        // std::cout << "det(R) = " << R.determinant() << "\n";
        Eigen::Matrix<double, 4, 2> tip_coords;

        for (int n = 0; n < 4; ++n)
        {
            tip_coords.row(n) = mesh.vertices[tip_element[n]];
        }

        const double xi_tip = tip_data.tip_point_local_coords.x();
        const double eta_tip = tip_data.tip_point_local_coords.y();

        std::array<double, 4> N_tip;
        N_tip[0] = 0.25 * (1.0 - xi_tip) * (1.0 - eta_tip);
        N_tip[1] = 0.25 * (1.0 + xi_tip) * (1.0 - eta_tip);
        N_tip[2] = 0.25 * (1.0 + xi_tip) * (1.0 + eta_tip);
        N_tip[3] = 0.25 * (1.0 - xi_tip) * (1.0 + eta_tip);

        Eigen::Vector2d tip_point_global = Eigen::Vector2d::Zero();

        for (int n = 0; n < 4; ++n)
        {
            tip_point_global += N_tip[n] * tip_coords.row(n).transpose();
        }

        const double h_tip = elementSize(tip_coords);

        const double Rin_h = Rin * h_tip;
        const double Rout_h = Rout * h_tip;

        std::cout << "Rin_h, Rout_h: " << Rin_h << " " << Rout_h << std::endl;

        double I_mode1 = 0.0;
        double I_mode2 = 0.0;
        int used_elements = 0;

        auto integrateAtPoint = [&](const std::array<int, 4> &element, const Eigen::Matrix<double, 4, 2> &coords,
                                    double xi, double eta, double integration_weight, int forced_H_value) {
            LinearQuad::ShapeData shape = fillQuadShape(xi, eta);

            LinearTriangle::JacobianData jd;
            jd.J = shape.dN_xi_eta * coords;

            bool invertible = false;
            jd.J.computeInverseAndDetWithCheck(jd.invJ, jd.detJ, invertible, 1e-12);

            if (!invertible)
            {
                throw std::runtime_error("Jacobi matrix is not invertible");
            }

            Eigen::Matrix<double, 2, 4> dN_dx_dy = jd.invJ * shape.dN_xi_eta;

            Eigen::Vector2d x_global = Eigen::Vector2d::Zero();

            for (int n = 0; n < 4; ++n)
            {
                x_global += shape.N[n] * coords.row(n).transpose();
            }

            const Eigen::Vector2d d = x_global - tip_point_global;

            const double x1 = d.dot(t_vec);
            const double x2 = d.dot(n_vec);

            const double radius2 = x1 * x1 + x2 * x2;
            const double radius = std::sqrt(radius2);

            if (radius < 1e-12 || radius >= Rout_h)
            {
                return;
            }

            const double theta = std::atan2(x2, x1);
            const double sqrt_r = std::sqrt(radius);
            const double inv_sqrt_r = 1.0 / sqrt_r;

            const double sin_half = std::sin(theta / 2.0);
            const double cos_half = std::cos(theta / 2.0);
            const double sin_theta = std::sin(theta);
            const double cos_theta = std::cos(theta);
            const double sin_three_half = std::sin(3.0 * theta / 2.0);
            const double cos_three_half = std::cos(3.0 * theta / 2.0);

            std::array<double, 4> f = {sqrt_r * sin_half, sqrt_r * cos_half, sqrt_r * sin_theta * sin_half,
                                       sqrt_r * sin_theta * cos_half};

            std::array<double, 4> df_dr;
            std::array<double, 4> df_dtheta;

            df_dr[0] = 0.5 * inv_sqrt_r * sin_half;
            df_dr[1] = 0.5 * inv_sqrt_r * cos_half;
            df_dr[2] = 0.5 * inv_sqrt_r * sin_half * sin_theta;
            df_dr[3] = 0.5 * inv_sqrt_r * cos_half * sin_theta;

            df_dtheta[0] = sqrt_r * 0.5 * cos_half;
            df_dtheta[1] = -sqrt_r * 0.5 * sin_half;
            df_dtheta[2] = sqrt_r * (0.5 * cos_half * sin_theta + sin_half * cos_theta);
            df_dtheta[3] = sqrt_r * (-0.5 * sin_half * sin_theta + cos_half * cos_theta);

            const double drdx = d.x() / radius;
            const double drdy = d.y() / radius;

            const double dtheta_dx = (x1 * n_vec.x() - x2 * t_vec.x()) / radius2;

            const double dtheta_dy = (x1 * n_vec.y() - x2 * t_vec.y()) / radius2;

            std::array<Eigen::Vector2d, 4> df_dx;

            for (int a = 0; a < 4; ++a)
            {
                df_dx[a].x() = df_dr[a] * drdx + df_dtheta[a] * dtheta_dx;
                df_dx[a].y() = df_dr[a] * drdy + df_dtheta[a] * dtheta_dy;
            }

            std::array<std::array<double, 4>, 4> f_nodes;

            for (int n = 0; n < 4; ++n)
            {
                Eigen::Vector2d d_node = coords.row(n).transpose() - tip_point_global;

                const double node_x1 = d_node.dot(t_vec);
                const double node_x2 = d_node.dot(n_vec);

                const double node_radius = std::sqrt(node_x1 * node_x1 + node_x2 * node_x2);

                const double node_theta = std::atan2(node_x2, node_x1);

                const double node_sqrt_r = std::sqrt(node_radius);
                const double node_sin_half = std::sin(node_theta / 2.0);
                const double node_cos_half = std::cos(node_theta / 2.0);
                const double node_sin_theta = std::sin(node_theta);

                f_nodes[n] = {node_sqrt_r * node_sin_half, node_sqrt_r * node_cos_half,
                              node_sqrt_r * node_sin_theta * node_sin_half,
                              node_sqrt_r * node_sin_theta * node_cos_half};
            }

            Eigen::Matrix2d grad_u_global;
            grad_u_global.setZero();

            int H_at_x = forced_H_value;

            if (H_at_x == 0)
            {
                H_at_x = (x2 >= 0.0) ? 1 : -1;
            }

            for (int n = 0; n < 4; ++n)
            {
                const int node = element[n];
                const int off = node_offset[node];

                const double dNdx = dN_dx_dy(0, n);
                const double dNdy = dN_dx_dy(1, n);
                const double Nn = shape.N[n];

                const double ux = u_solu(off);
                const double uy = u_solu(off + 1);

                grad_u_global(0, 0) += dNdx * ux;
                grad_u_global(0, 1) += dNdy * ux;
                grad_u_global(1, 0) += dNdx * uy;
                grad_u_global(1, 1) += dNdy * uy;

                if (heaviside_enriched_nodes[node])
                {
                    const int H_i = vertices_level_set_signs[node].sign;
                    const double H_shift = static_cast<double>(H_at_x - H_i);

                    const double ax = u_solu(off + 2);
                    const double ay = u_solu(off + 3);

                    grad_u_global(0, 0) += dNdx * H_shift * ax;
                    grad_u_global(0, 1) += dNdy * H_shift * ax;
                    grad_u_global(1, 0) += dNdx * H_shift * ay;
                    grad_u_global(1, 1) += dNdy * H_shift * ay;
                }

                if (tip_enriched_nodes[node])
                {
                    for (int a = 0; a < 4; ++a)
                    {
                        const double bx = u_solu(off + 4 + 2 * a);
                        const double by = u_solu(off + 4 + 2 * a + 1);

                        const double shift = f[a] - f_nodes[n][a];

                        const double d_enr_dx = dNdx * shift + Nn * df_dx[a].x();

                        const double d_enr_dy = dNdy * shift + Nn * df_dx[a].y();

                        grad_u_global(0, 0) += d_enr_dx * bx;
                        grad_u_global(0, 1) += d_enr_dy * bx;
                        grad_u_global(1, 0) += d_enr_dx * by;
                        grad_u_global(1, 1) += d_enr_dy * by;
                    }
                }
            }

            Eigen::Vector3d strain_global;
            strain_global << grad_u_global(0, 0), grad_u_global(1, 1), grad_u_global(0, 1) + grad_u_global(1, 0);

            const Eigen::Vector3d stress_global_voigt = D * strain_global;

            Eigen::Matrix2d sigma_global;
            sigma_global << stress_global_voigt(0), stress_global_voigt(2), stress_global_voigt(2),
                stress_global_voigt(1);

            const Eigen::Matrix2d sigma_local = R.transpose() * sigma_global * R;

            const Eigen::Matrix2d grad_u_local = R.transpose() * grad_u_global * R;

            Eigen::Vector3d stress;
            stress << sigma_local(0, 0), sigma_local(1, 1), sigma_local(0, 1);

            const double du1_dx1_act = grad_u_local(0, 0);
            const double du2_dx1_act = grad_u_local(1, 0);

            const double inv_sqrt_2pi_inv_sqrt_r = INV_SQRT_2PI * inv_sqrt_r;

            Eigen::Vector3d sigma_aux_mode_1;
            sigma_aux_mode_1 << cos_half * (1.0 - sin_half * sin_three_half),
                cos_half * (1.0 + sin_half * sin_three_half), sin_half * cos_half * cos_three_half;

            sigma_aux_mode_1 *= inv_sqrt_2pi_inv_sqrt_r;

            Eigen::Vector3d sigma_aux_mode_2;
            sigma_aux_mode_2 << -sin_half * (2.0 + cos_half * cos_three_half), sin_half * cos_half * cos_three_half,
                cos_half * (1.0 - sin_half * sin_three_half);

            sigma_aux_mode_2 *= inv_sqrt_2pi_inv_sqrt_r;

            auto auxDisplacement = [&](double xx1, double xx2, int mode) -> Eigen::Vector2d {
                const double rr2 = xx1 * xx1 + xx2 * xx2;
                const double rr = std::sqrt(rr2);

                if (rr < 1e-14)
                {
                    return Eigen::Vector2d::Zero();
                }

                const double th = std::atan2(xx2, xx1);
                const double sqrt_rr = std::sqrt(rr);

                const double sh = std::sin(th / 2.0);
                const double ch = std::cos(th / 2.0);
                const double cth = std::cos(th);

                const double factor = (1.0 / (2.0 * shear_modulus)) * INV_SQRT_2PI * sqrt_rr;

                if (mode == 1)
                {
                    // Mode I auxiliary displacement, K_I_aux = 1
                    return factor *
                           Eigen::Vector2d{ch * (kappa - 1.0 + 2.0 * sh * sh), sh * (kappa + 1.0 - 2.0 * ch * ch)};
                }

                // Mode II auxiliary displacement, K_II_aux = 1
                return factor * Eigen::Vector2d{sh * (kappa + 2.0 + cth), -ch * (kappa - 2.0 + cth)};
            };

            // Analytical derivatives of auxiliary displacement with respect to local x1.
            //
            // u_aux = C * sqrt(r) * G(theta)
            // C = 1 / (2 * mu * sqrt(2*pi))
            //
            // d/dx1 [C * sqrt(r) * G(theta)]
            // = C / sqrt(r) * [0.5 * cos(theta) * G(theta)
            //                  - sin(theta) * dG/dtheta]
            //
            // because:
            // dr/dx1 = cos(theta)
            // dtheta/dx1 = -sin(theta) / r

            const double aux_deriv_factor = (1.0 / (2.0 * shear_modulus)) * INV_SQRT_2PI * inv_sqrt_r;

            // --------------------
            // Mode I auxiliary field
            // --------------------
            //
            // u1_I = C * sqrt(r) * cos(theta/2) * (kappa - cos(theta))
            // u2_I = C * sqrt(r) * sin(theta/2) * (kappa - cos(theta))

            const double A_I = kappa - cos_theta;

            const double G1_I = cos_half * A_I;
            const double G2_I = sin_half * A_I;

            // d/dtheta(kappa - cos(theta)) = sin(theta)
            const double dG1_I_dtheta = -0.5 * sin_half * A_I + cos_half * sin_theta;

            const double dG2_I_dtheta = 0.5 * cos_half * A_I + sin_half * sin_theta;

            const double du1_dx1_aux_mode_1 = aux_deriv_factor * (0.5 * cos_theta * G1_I - sin_theta * dG1_I_dtheta);

            const double du2_dx1_aux_mode_1 = aux_deriv_factor * (0.5 * cos_theta * G2_I - sin_theta * dG2_I_dtheta);

            // --------------------
            // Mode II auxiliary field
            // --------------------
            //
            // u1_II = C * sqrt(r) * sin(theta/2) * (kappa + 2 + cos(theta))
            // u2_II = -C * sqrt(r) * cos(theta/2) * (kappa - 2 + cos(theta))

            const double A_II_1 = kappa + 2.0 + cos_theta;
            const double A_II_2 = kappa - 2.0 + cos_theta;

            const double G1_II = sin_half * A_II_1;
            const double G2_II = -cos_half * A_II_2;

            // d/dtheta(kappa + 2 + cos(theta)) = -sin(theta)
            // d/dtheta(kappa - 2 + cos(theta)) = -sin(theta)

            const double dG1_II_dtheta = 0.5 * cos_half * A_II_1 - sin_half * sin_theta;

            const double dG2_II_dtheta = 0.5 * sin_half * A_II_2 + cos_half * sin_theta;

            const double du1_dx1_aux_mode_2 = aux_deriv_factor * (0.5 * cos_theta * G1_II - sin_theta * dG1_II_dtheta);

            const double du2_dx1_aux_mode_2 = aux_deriv_factor * (0.5 * cos_theta * G2_II - sin_theta * dG2_II_dtheta);

            const double eps11_aux1 = invE * (sigma_aux_mode_1(0) - nu * sigma_aux_mode_1(1));

            const double eps22_aux1 = invE * (sigma_aux_mode_1(1) - nu * sigma_aux_mode_1(0));

            const double eps12_aux1 = (1.0 + nu) / E * sigma_aux_mode_1(2);

            const double eps11_aux2 = invE * (sigma_aux_mode_2(0) - nu * sigma_aux_mode_2(1));

            const double eps22_aux2 = invE * (sigma_aux_mode_2(1) - nu * sigma_aux_mode_2(0));

            const double eps12_aux2 = (1.0 + nu) / E * sigma_aux_mode_2(2);

            const double W12_mode1 = stress(0) * eps11_aux1 + stress(1) * eps22_aux1 + 2.0 * stress(2) * eps12_aux1;

            const double W12_mode2 = stress(0) * eps11_aux2 + stress(1) * eps22_aux2 + 2.0 * stress(2) * eps12_aux2;

            const double A1_mode1 = stress(0) * du1_dx1_aux_mode_1 + stress(2) * du2_dx1_aux_mode_1 +
                                    sigma_aux_mode_1(0) * du1_dx1_act + sigma_aux_mode_1(2) * du2_dx1_act - W12_mode1;

            const double A2_mode1 = stress(2) * du1_dx1_aux_mode_1 + stress(1) * du2_dx1_aux_mode_1 +
                                    sigma_aux_mode_1(2) * du1_dx1_act + sigma_aux_mode_1(1) * du2_dx1_act;

            const double A1_mode2 = stress(0) * du1_dx1_aux_mode_2 + stress(2) * du2_dx1_aux_mode_2 +
                                    sigma_aux_mode_2(0) * du1_dx1_act + sigma_aux_mode_2(2) * du2_dx1_act - W12_mode2;

            const double A2_mode2 = stress(2) * du1_dx1_aux_mode_2 + stress(1) * du2_dx1_aux_mode_2 +
                                    sigma_aux_mode_2(2) * du1_dx1_act + sigma_aux_mode_2(1) * du2_dx1_act;

            double dqdr = 0.0;

            if (radius <= Rin_h)
            {
                dqdr = 0.0;
            }
            else if (radius >= Rout_h)
            {
                dqdr = 0.0;
            }
            else
            {
                dqdr = -1.0 / (Rout_h - Rin_h);
            }

            const double dqdx1 = dqdr * x1 / radius;
            const double dqdx2 = dqdr * x2 / radius;

            const double integrand_mode1 = A1_mode1 * dqdx1 + A2_mode1 * dqdx2;

            const double integrand_mode2 = A1_mode2 * dqdx1 + A2_mode2 * dqdx2;

            I_mode1 += integrand_mode1 * integration_weight;
            I_mode2 += integrand_mode2 * integration_weight;
        };

        for (size_t elem_id = 0; elem_id < mesh.elements.size(); ++elem_id)
        {
            const std::array<int, 4> &element = mesh.elements[elem_id];

            Eigen::Matrix<double, 4, 2> coords;

            for (int n = 0; n < 4; ++n)
            {
                coords.row(n) = mesh.vertices[element[n]];
            }

            if (!elementIntersectsRadius(coords, tip_point_global, Rout_h))
            {
                continue;
            }

            used_elements++;

            const int elem_id_int = static_cast<int>(elem_id);
            const int heaviside_index = heaviside_element_to_index[elem_id_int];

            // ------------------------------------------------------------
            // 1. Current crack-tip element: use tip triangulation
            // ------------------------------------------------------------
            if (elem_id_int == tip_data.id)
            {
                const TipTriangulation &triangulation = tip_enriched_triangulation[tip_idx];

                std::array<Eigen::Vector2d, 6> local_points = {Eigen::Vector2d{-1.0, -1.0},
                                                               Eigen::Vector2d{1.0, -1.0},
                                                               Eigen::Vector2d{1.0, 1.0},
                                                               Eigen::Vector2d{-1.0, 1.0},
                                                               tip_data.intersection_point_local_coords,
                                                               tip_data.tip_point_local_coords};

                for (unsigned int tri_id = 0; tri_id < 5; ++tri_id)
                {
                    const std::array<unsigned char, 3> &triangle = triangulation.tri_indices[tri_id];

                    Eigen::Matrix2d J_xieta_rs;
                    J_xieta_rs << local_points[triangle[1]].x() - local_points[triangle[0]].x(),
                        local_points[triangle[2]].x() - local_points[triangle[0]].x(),
                        local_points[triangle[1]].y() - local_points[triangle[0]].y(),
                        local_points[triangle[2]].y() - local_points[triangle[0]].y();

                    const double det_tri = J_xieta_rs.determinant();

                    for (unsigned int gp = 0; gp < NGauss; ++gp)
                    {
                        const double r_tri = gauss_pts[gp][0];
                        const double s_tri = gauss_pts[gp][1];
                        const double t_tri = 1.0 - r_tri - s_tri;

                        const double xi = local_points[triangle[0]].x() * t_tri +
                                          local_points[triangle[1]].x() * r_tri + local_points[triangle[2]].x() * s_tri;

                        const double eta = local_points[triangle[0]].y() * t_tri +
                                           local_points[triangle[1]].y() * r_tri +
                                           local_points[triangle[2]].y() * s_tri;

                        LinearQuad::ShapeData tmp_shape = fillQuadShape(xi, eta);

                        LinearTriangle::JacobianData tmp_jd;
                        tmp_jd.J = tmp_shape.dN_xi_eta * coords;

                        bool tmp_invertible = false;
                        tmp_jd.J.computeInverseAndDetWithCheck(tmp_jd.invJ, tmp_jd.detJ, tmp_invertible, 1e-12);

                        if (!tmp_invertible)
                        {
                            throw std::runtime_error("Jacobi matrix is not invertible");
                        }

                        const double weight = gauss_wts[gp] * std::abs(det_tri) * std::abs(tmp_jd.detJ);

                        // forced_H_value = 0:
                        // inside tip element we let H be determined by local x2
                        integrateAtPoint(element, coords, xi, eta, weight, 0);
                    }
                }
            }

            // ------------------------------------------------------------
            // 2. Heaviside-enriched element inside the patch:
            //    use Heaviside triangulation, not regular 2x2 quadrature
            // ------------------------------------------------------------
            else if (heaviside_index >= 0)
            {
                const HeavisideEnriched &h_data = heaviside_enriched[heaviside_index];

                const HeavisideTriangulation &h_triangulation = heaviside_enriched_triangulation[heaviside_index];

                std::array<Eigen::Vector2d, 6> local_points = {Eigen::Vector2d{-1.0, -1.0},
                                                               Eigen::Vector2d{1.0, -1.0},
                                                               Eigen::Vector2d{1.0, 1.0},
                                                               Eigen::Vector2d{-1.0, 1.0},
                                                               h_data.intersection_points_local_coords[0],
                                                               h_data.intersection_points_local_coords[1]};

                for (unsigned int tri_id = 0; tri_id < h_triangulation.triangles_num; ++tri_id)
                {
                    const std::array<unsigned char, 3> &triangle = h_triangulation.tri_indices[tri_id];

                    const int forced_H_value = tri_id < h_triangulation.positive_heaviside_triangles_num ? 1 : -1;

                    Eigen::Matrix2d J_xieta_rs;
                    J_xieta_rs << local_points[triangle[1]].x() - local_points[triangle[0]].x(),
                        local_points[triangle[2]].x() - local_points[triangle[0]].x(),
                        local_points[triangle[1]].y() - local_points[triangle[0]].y(),
                        local_points[triangle[2]].y() - local_points[triangle[0]].y();

                    const double det_tri = J_xieta_rs.determinant();

                    for (unsigned int gp = 0; gp < NGauss; ++gp)
                    {
                        const double r_tri = gauss_pts[gp][0];
                        const double s_tri = gauss_pts[gp][1];
                        const double t_tri = 1.0 - r_tri - s_tri;

                        const double xi = local_points[triangle[0]].x() * t_tri +
                                          local_points[triangle[1]].x() * r_tri + local_points[triangle[2]].x() * s_tri;

                        const double eta = local_points[triangle[0]].y() * t_tri +
                                           local_points[triangle[1]].y() * r_tri +
                                           local_points[triangle[2]].y() * s_tri;

                        LinearQuad::ShapeData tmp_shape = fillQuadShape(xi, eta);

                        LinearTriangle::JacobianData tmp_jd;
                        tmp_jd.J = tmp_shape.dN_xi_eta * coords;

                        bool tmp_invertible = false;
                        tmp_jd.J.computeInverseAndDetWithCheck(tmp_jd.invJ, tmp_jd.detJ, tmp_invertible, 1e-12);

                        if (!tmp_invertible)
                        {
                            throw std::runtime_error("Jacobi matrix is not invertible");
                        }

                        const double weight = gauss_wts[gp] * std::abs(det_tri) * std::abs(tmp_jd.detJ);

                        integrateAtPoint(element, coords, xi, eta, weight, forced_H_value);
                    }
                }
            }

            // ------------------------------------------------------------
            // 3. Regular element inside the patch:
            //    use standard 2x2 quadrature
            // ------------------------------------------------------------
            else
            {
                const double g = 1.0 / std::sqrt(3.0);
                const std::array<double, 2> gp_1d = {-g, g};

                for (double xi : gp_1d)
                {
                    for (double eta : gp_1d)
                    {
                        LinearQuad::ShapeData tmp_shape = fillQuadShape(xi, eta);

                        LinearTriangle::JacobianData tmp_jd;
                        tmp_jd.J = tmp_shape.dN_xi_eta * coords;

                        bool tmp_invertible = false;
                        tmp_jd.J.computeInverseAndDetWithCheck(tmp_jd.invJ, tmp_jd.detJ, tmp_invertible, 1e-12);

                        if (!tmp_invertible)
                        {
                            throw std::runtime_error("Jacobi matrix is not invertible");
                        }

                        const double weight = std::abs(tmp_jd.detJ);

                        integrateAtPoint(element, coords, xi, eta, weight, 0);
                    }
                }
            }
        }

        const double E_prime = E; // plane stress
        // plane strain:
        // const double E_prime = E / (1.0 - nu * nu);

        const double K_I = 0.5 * E_prime * I_mode1;
        const double K_II = 0.5 * E_prime * I_mode2;

        std::cout << "tip_index = " << static_cast<int>(tip_data.tip_index) << ", used_elements = " << used_elements
                  << ", K_I = " << K_I << ", K_II = " << K_II
                  << ", |KII/KI| = " << (std::abs(K_I) > 1e-14 ? std::abs(K_II / K_I) : 0.0) << std::endl;
    }
}

// It turns out that for templates we need to either implement a function in a header file or explicitly create a
// template as shown below
template void computeStress<13>(
    const std::vector<TipEnriched> &tip_enriched, const std::vector<HeavisideEnriched> &heaviside_enriched,
    const QuadMesh &mesh, const std::vector<TipTriangulation> &tip_enriched_triangulation,
    const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation, const Eigen::VectorXd &u_solu,
    const std::vector<unsigned int> &node_offset, const std::vector<bool> &heaviside_enriched_nodes,
    const std::vector<bool> &tip_enriched_nodes, const std::vector<LevelSetSign> &vertices_level_set_signs,
    const std::array<std::array<double, 2>, 13> &gauss_pts, const std::array<double, 13> &gauss_wts,
    const Eigen::Vector2d &crack_tip_1_t, const Eigen::Vector2d &crack_tip_1_n, const Eigen::Vector2d &crack_tip_2_t,
    const Eigen::Vector2d &crack_tip_2_n, const Eigen::Matrix3d &D, const double young_modulus,
    const double poisson_ratio, const double Rin, const double Rout);