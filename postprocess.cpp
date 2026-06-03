#include "postprocess.h"

#include "misc.h"

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
        std::cout << u << std::endl;
        std::cout << a << std::endl;
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

void drawTipElements(const std::vector<TipEnriched> &tip_enriched, const QuadMesh &mesh,
                     const std::vector<TipTriangulation> &tip_enriched_triangulation,
                     const std::vector<LevelSetSign> &vertices_level_set_signs, const Eigen::VectorXd &u_solu,
                     const std::vector<unsigned int> &node_offset, const std::vector<bool> &tip_enriched_nodes,
                     double scale, std::vector<PolygonalChain> &polygonal_chains,
                     const Eigen::Vector2d &crack_tip_1_t, const Eigen::Vector2d &crack_tip_1_n,
                     const Eigen::Vector2d &crack_tip_2_t, const Eigen::Vector2d &crack_tip_2_n)
{
    for (int i = 0; i < tip_enriched.size(); i++)
    {
        const TipEnriched &tip_enr = tip_enriched[i];
        const std::array<int, 4> &element = mesh.elements[tip_enr.id];
        // const TipTriangulation& tip_trng = tip_enriched_triangulation[i];
        constexpr int N_points = 100;
        const Eigen::Vector2d tip_point_global_coords =
            mesh.vertices[element[0]] * (1 - tip_enr.tip_point_local_coords.x()) * (1 - tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[1]] * (1 + tip_enr.tip_point_local_coords.x()) * (1 - tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[2]] * (1 + tip_enr.tip_point_local_coords.x()) * (1 + tip_enr.tip_point_local_coords.y()) / 4 +
            mesh.vertices[element[3]] * (1 - tip_enr.tip_point_local_coords.x()) * (1 + tip_enr.tip_point_local_coords.y()) / 4;
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
        const Eigen::Vector2d crack_dir_local = tip_enr.tip_point_local_coords - tip_enr.intersection_point_local_coords;
        const double L = crack_dir_local.norm();
        Eigen::Matrix<double, 4, 2> u;
        u.row(0) = u_solu.segment<2>(node_offset[element[0]]) * scale;
        u.row(1) = u_solu.segment<2>(node_offset[element[1]]) * scale;
        u.row(2) = u_solu.segment<2>(node_offset[element[2]]) * scale;
        u.row(3) = u_solu.segment<2>(node_offset[element[3]]) * scale;
        std::array<Eigen::Matrix<double,4,2>,4> b_nodes; // a_nodes[n][a][component]
        for (int n=0; n<4; ++n) {
            if (!tip_enriched_nodes[element[n]])
            {
                throw std::runtime_error("NotImplemented: not tip enriched node!!!");
            }
            int base = node_offset[element[n]];
            for (int a=0; a<4; ++a) {
                b_nodes[n](a,0) = u_solu(base + 2 + 2 + 2*a) * scale;
                b_nodes[n](a,1) = u_solu(base + 2 + 2 + 2*a + 1) * scale;
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
            double xi = tip_enr.intersection_point_local_coords.x() + crack_dir_local.x() * i / (N_points ) ;
            double eta = tip_enr.intersection_point_local_coords.y() + crack_dir_local.y() * i / (N_points );
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
            for (int n = 0; n < 4; n++){
                for (int a = 0; a < 4; a++){
                    p0 += N[n]*(f[a] - f_nodes[n][a])*b_nodes[n].row(a);
                }
            }
            upper_points.push_back(glm::vec2{static_cast<float>(p0.x()),static_cast<float>(p0.y())});
        }
        for (int i = N_points; i >= 0; i--)
        {
            double xi = tip_enr.intersection_point_local_coords.x() + crack_dir_local.x() * i / (N_points ) ;
            double eta = tip_enr.intersection_point_local_coords.y() + crack_dir_local.y() * i / (N_points );
            const Eigen::RowVector4d N{(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, (1 + xi) * (1 + eta) / 4,
                                    (1 - xi) * (1 + eta) / 4};
            const Eigen::RowVector2d p = N * vertices;
            Eigen::Vector2d d_phys = p.transpose() - tip_point_global_coords;
            double r = d_phys.norm();
            Eigen::Vector2d to_intersection = intersection_point_global_coords - tip_point_global_coords;

            double check = to_intersection.normalized().dot(crack_tip_1_t.normalized());
            std::cout << "orientation check = " << check << std::endl;
            const double sqrt_r = std::sqrt(r);
            const double sinhalftheta = -1;
            const double sintheta = 0;
            const double coshalftheta = 0;
            const std::array<double, 4> f = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta,
                                             sqrt_r * sintheta * sinhalftheta, sqrt_r * sintheta * coshalftheta};

            Eigen::RowVector2d p0 = p + N * u;
            for (int n = 0; n < 4; n++){
                for (int a = 0; a < 4; a++){
                    p0 += N[n]*(f[a] - f_nodes[n][a])*b_nodes[n].row(a);
                }
            }
            lower_points.push_back(glm::vec2{static_cast<float>(p0.x()),static_cast<float>(p0.y())});
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
