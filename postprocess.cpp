#include "postprocess.h"

#include "misc.h"
#include "fem.h"

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

void drawTipElements(const std::vector<TipEnriched> &tip_enriched, const QuadMesh &mesh,
                     const Eigen::VectorXd &u_solu,
                     const std::vector<unsigned int> &node_offset, 
                     const std::vector<bool> &heaviside_enriched_nodes,
                     const std::vector<bool> &tip_enriched_nodes,
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
            // int offset = 2;
            // if (heaviside_enriched_nodes[element[n]]) offset += 2;
            int offset = 4;
            for (int a=0; a<4; ++a) {
                b_nodes[n](a,0) = u_solu(base + offset + 2*a) * scale;
                b_nodes[n](a,1) = u_solu(base + offset + 2*a + 1) * scale;
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

template <unsigned int NGauss>
void computeStress(const std::vector<TipEnriched> &tip_enriched, const QuadMesh &mesh,
                   const std::vector<TipTriangulation> &tip_enriched_triangulation, const Eigen::VectorXd &u_solu,
                   const std::vector<unsigned int> &node_offset,
                   const std::vector<bool> &heaviside_enriched_nodes,
                   const std::array<std::array<double,2>, NGauss>& gauss_pts,
                   const std::array<double, NGauss>& gauss_wts,
                   const Eigen::Vector2d &crack_tip_1_t, const Eigen::Vector2d &crack_tip_1_n,
                   const Eigen::Vector2d &crack_tip_2_t, const Eigen::Vector2d &crack_tip_2_n,
                   const Eigen::Matrix3d& D)
{
    for (size_t i = 0; i < tip_enriched.size(); i++)
    {
        const TipEnriched &enriched_element = tip_enriched[i];
        const std::array<int, 4> &element = mesh.elements[enriched_element.id];
        const TipTriangulation &triangulation = tip_enriched_triangulation[i];
        // u_x u_y f1_x f1_y f2_x f2_y f3_x f3_y f4_x f4_y . total 10 dof per node
        // 4 nodes. total 40 dofs per element
        std::array<Eigen::Vector2d, 4> points = {
            mesh.vertices[element[0]],
            mesh.vertices[element[1]],
            mesh.vertices[element[2]],
            mesh.vertices[element[3]],
        };
        std::array<Eigen::Vector2d, 6> local_points = {Eigen::Vector2d{-1, -1},
                                                       Eigen::Vector2d{1, -1},
                                                       Eigen::Vector2d{1, 1},
                                                       Eigen::Vector2d{-1, 1},
                                                       enriched_element.intersection_point_local_coords,
                                                       enriched_element.tip_point_local_coords};
        const Eigen::Matrix<double, 4, 2> coords{{{points[0].x(), points[0].y()},
                                                  {points[1].x(), points[1].y()},
                                                  {points[2].x(), points[2].y()},
                                                  {points[3].x(), points[3].y()}}};
        const int ndof_per_node = 10;               // standard (2) + tip (8)
        Eigen::VectorXd u_e(4 * ndof_per_node);     // size 40

        for (int i = 0; i < 4; ++i) {
            int node = element[i];
            int off = node_offset[node];            // global start index for this node
            
            // Standard displacements (indices 0,1) -> positions 0,1 in u_e
            u_e.segment(ndof_per_node * i, 2) = u_solu.segment(off, 2);
            
            // Skip Heaviside DOFs (indices 2,3) – nothing copied
            
            // Tip enrichment DOFs (indices 4..11) -> positions 2..9 in u_e
            u_e.segment(ndof_per_node * i + 2, 8) = u_solu.segment(off + 4, 8);
        }
        std::array<std::array<double, 4>, 4> f_nodes;
        Eigen::Vector2d d;
        double radius, radius2, theta, sqrt_r, sinhalftheta, sintheta, coshalftheta, costheta;
        for (int n = 0; n < 4; n++)
        {
            Eigen::Vector2d tip_point_global_coords = Eigen::Vector2d::Zero();
            double xi_tip  = enriched_element.tip_point_local_coords.x();
            double eta_tip = enriched_element.tip_point_local_coords.y();
            std::array<double,4> N_tip;
            
            N_tip[0] = 0.25 * (1 - xi_tip) * (1 - eta_tip);
            N_tip[1] = 0.25 * (1 + xi_tip) * (1 - eta_tip);
            N_tip[2] = 0.25 * (1 + xi_tip) * (1 + eta_tip);
            N_tip[3] = 0.25 * (1 - xi_tip) * (1 + eta_tip);
            for (int k = 0; k < 4; k++)
            {
                tip_point_global_coords += N_tip[k] * coords.row(k);
            }
            d = (coords.row(n).transpose() - tip_point_global_coords);
            radius = d.norm();

            if (enriched_element.tip_index == 1){
                theta = std::atan2(d.dot(crack_tip_1_n), d.dot(crack_tip_1_t)) ;
            }
            else if (enriched_element.tip_index == 2){
                theta = std::atan2(d.dot(crack_tip_2_n), d.dot(crack_tip_2_t)) ;
            }
            
            sqrt_r = std::sqrt(radius);
            sinhalftheta = std::sin(theta / 2);
            sintheta = std::sin(theta);
            coshalftheta = std::cos(theta / 2);
            f_nodes[n] = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta, sqrt_r * sintheta * sinhalftheta,
                    sqrt_r * sintheta * coshalftheta};
        }
        std::array<double, 4> dfdr, dfdtheta;
        double drdx, drdy, dthetadx, dthetady;
        std::array<Eigen::Vector2d,4> df_dx;
        double dNdx, dNdy, Nn;
        double shift;
        double factor;
        Eigen::Matrix<double, 3, 40> B;

        double total_area = 0.0;

        for (unsigned int j = 0; j < 5; j++)
        {

            const std::array<unsigned char, 3> &triangle = triangulation.tri_indices[j];

            Eigen::Matrix2d J_xieta_rs{{{local_points[triangle[1]].x() - local_points[triangle[0]].x(),
                                         local_points[triangle[2]].x() - local_points[triangle[0]].x()},
                                        {local_points[triangle[1]].y() - local_points[triangle[0]].y(),
                                         local_points[triangle[2]].y() - local_points[triangle[0]].y()}}};
            double det_tri = J_xieta_rs.determinant();
            for (unsigned int gp = 0; gp < NGauss; gp++)
            {
                B.setZero();
                double r = gauss_pts[gp][0];
                double s = gauss_pts[gp][1];
                double t = 1 - r - s;
                double xi = local_points[triangle[0]].x() * t + local_points[triangle[1]].x() * r +
                           local_points[triangle[2]].x() * s;
                double eta = local_points[triangle[0]].y() * t + local_points[triangle[1]].y() * r +
                            local_points[triangle[2]].y() * s;
                LinearQuad::ShapeData shape;
                // Node 1
                shape.N[0] = 0.25 * (1 - xi) * (1 - eta);
                shape.dN_xi_eta(0, 0) = -0.25 * (1 - eta);
                shape.dN_xi_eta(1, 0) = -0.25 * (1 - xi);
                // Node 2
                shape.N[1] = 0.25 * (1 + xi) * (1 - eta);
                shape.dN_xi_eta(0, 1) = 0.25 * (1 - eta);
                shape.dN_xi_eta(1, 1) = -0.25 * (1 + xi);
                // Node 3
                shape.N[2] = 0.25 * (1 + xi) * (1 + eta);
                shape.dN_xi_eta(0, 2) = 0.25 * (1 + eta);
                shape.dN_xi_eta(1, 2) = 0.25 * (1 + xi);
                // Node 4
                shape.N[3] = 0.25 * (1 - xi) * (1 + eta);
                shape.dN_xi_eta(0, 3) = -0.25 * (1 + eta);
                shape.dN_xi_eta(1, 3) = 0.25 * (1 - xi);

                LinearTriangle::JacobianData jd;
                jd.J = shape.dN_xi_eta * coords;
                bool invertible;
                jd.J.computeInverseAndDetWithCheck(jd.invJ, jd.detJ, invertible, 1e-12);
                if (!invertible)
                    throw std::runtime_error("Jacobi matrix is not invertible");

                Eigen::Matrix<double, 2, 4> dN_dx_dy;
                dN_dx_dy = jd.invJ * shape.dN_xi_eta;

                Eigen::Vector2d gauss_point_global_coords = Eigen::Vector2d::Zero();
                Eigen::Vector2d tip_point_global_coords = Eigen::Vector2d::Zero();
                double xi_tip  = enriched_element.tip_point_local_coords.x();
                double eta_tip = enriched_element.tip_point_local_coords.y();
                std::array<double,4> N_tip;
                
                N_tip[0] = 0.25 * (1 - xi_tip) * (1 - eta_tip);
                N_tip[1] = 0.25 * (1 + xi_tip) * (1 - eta_tip);
                N_tip[2] = 0.25 * (1 + xi_tip) * (1 + eta_tip);
                N_tip[3] = 0.25 * (1 - xi_tip) * (1 + eta_tip);
                for (int k = 0; k < 4; k++)
                {
                    gauss_point_global_coords += shape.N[k] * coords.row(k);
                    tip_point_global_coords += N_tip[k] * coords.row(k);
                }
                d = gauss_point_global_coords - tip_point_global_coords;
                radius2 = d.squaredNorm();
                radius = std::sqrt(radius2);
                if (enriched_element.tip_index == 1){
                    theta = std::atan2(d.dot(crack_tip_1_n), d.dot(crack_tip_1_t)) ;
                }
                else if (enriched_element.tip_index == 2){
                    theta = std::atan2(d.dot(crack_tip_2_n), d.dot(crack_tip_2_t)) ;
                }

                sqrt_r = std::sqrt(radius);
                sinhalftheta = std::sin(theta / 2);
                sintheta = std::sin(theta);
                coshalftheta = std::cos(theta / 2);
                costheta = std::cos(theta);
                std::array<double, 4> f = {sqrt_r * sinhalftheta, sqrt_r * coshalftheta,
                                           sqrt_r * sintheta * sinhalftheta, sqrt_r * sintheta * coshalftheta};

                dfdr[0] = 0.5 / sqrt_r * sinhalftheta;               // ∂f1/∂r
                dfdr[1] = 0.5 / sqrt_r * coshalftheta;
                dfdr[2] = 0.5 / sqrt_r * sinhalftheta * sintheta;
                dfdr[3] = 0.5 / sqrt_r * coshalftheta * sintheta;

                dfdtheta[0] = sqrt_r * 0.5 * coshalftheta;           // ∂f1/∂θ
                dfdtheta[1] = -sqrt_r * 0.5 * sinhalftheta;
                dfdtheta[2] = sqrt_r * (0.5 * coshalftheta * sintheta + sinhalftheta * costheta);
                dfdtheta[3] = sqrt_r * (-0.5 * sinhalftheta * sintheta + coshalftheta * costheta);

                drdx = (radius > 1e-12) ? d.x() / radius : 0.0;
                drdy = (radius > 1e-12) ? d.y() / radius : 0.0;
                if (enriched_element.tip_index == 1){
                    double a = d.dot(crack_tip_1_t);   // distance along tangent
                    double b = d.dot(crack_tip_1_n);   // distance along normal
                    double r2 = a*a + b*b;
                    if (r2 > 1e-12) {
                        dthetadx = (a * crack_tip_1_n.x() - b * crack_tip_1_t.x()) / r2;
                        dthetady = (a * crack_tip_1_n.y() - b * crack_tip_1_t.y()) / r2;
                    } else {
                        dthetadx = dthetady = 0.0;
                    }
                }
                else if (enriched_element.tip_index == 2){
                    double a = d.dot(crack_tip_2_t);   // distance along tangent
                    double b = d.dot(crack_tip_2_n);   // distance along normal
                    double r2 = a*a + b*b;
                    if (r2 > 1e-12) {
                        dthetadx = (a * crack_tip_2_n.x() - b * crack_tip_2_t.x()) / r2;
                        dthetady = (a * crack_tip_2_n.y() - b * crack_tip_2_t.y()) / r2;
                    } else {
                        dthetadx = dthetady = 0.0;
                    }
                }

                for (int a = 0; a < 4; ++a) {
                    df_dx[a].x() = dfdr[a] * drdx + dfdtheta[a] * dthetadx;
                    df_dx[a].y() = dfdr[a] * drdy + dfdtheta[a] * dthetady;
                }

                for (int n = 0; n < 4; ++n)
                {
                    dNdx = dN_dx_dy(0,n);
                    dNdy = dN_dx_dy(1,n);
                    Nn = shape.N[n];
                    // no enrnchment[edge]
                    B(0, 10 * n) = dNdx; // du/dx
                    B(0, 10 * n + 1) = 0;
                    B(1, 10 * n) = 0;
                    B(1, 10 * n + 1) = dNdy; // dv/dy
                    B(2, 10 * n) = dNdy;     // dv/dx
                    B(2, 10 * n + 1) = dNdx; // du/dy

                    // f_alpha
                    for (int a = 0; a < 4; a++)
                    {
                        shift = f[a] - f_nodes[n][a];
                        B(0, 10 * n + 2 + 2 * a) = dNdx * shift + Nn * df_dx[a].x(); // du/dx
                        B(0, 10 * n + 3 + 2 * a) = 0;
                        B(1, 10 * n + 2 + 2 * a) = 0;
                        B(1, 10 * n + 3 + 2 * a) = dNdy * shift + Nn * df_dx[a].y(); // dv/dy
                        B(2, 10 * n + 2 + 2 * a) = dNdy * shift + Nn * df_dx[a].y(); // dv/dx
                        B(2, 10 * n + 3 + 2 * a) = dNdx * shift + Nn * df_dx[a].x(); // du/dy
                    }
                }
                Eigen::Vector3d strain = B*u_e;
                std::cout << "Strain: " << strain << std::endl;
                Eigen::Vector3d stress = D*B*u_e;
                std::cout << "Stress:" << stress << std::endl;
                factor = gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
                if (det_tri < 0)
                    std::cout << "det_tri < 0" << std::endl;
                if (jd.detJ < 0)
                    std::cout << "jd.detJ < 0" << std::endl;
                total_area += gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
            }
        }
    }
}

template void computeStress<13>(const std::vector<TipEnriched> &tip_enriched, const QuadMesh &mesh,
                   const std::vector<TipTriangulation> &tip_enriched_triangulation, const Eigen::VectorXd &u_solu,
                   const std::vector<unsigned int> &node_offset,
                   const std::vector<bool> &heaviside_enriched_nodes,
                   const std::array<std::array<double,2>, 13>& gauss_pts,
                   const std::array<double, 13>& gauss_wts,
                   const Eigen::Vector2d &crack_tip_1_t, const Eigen::Vector2d &crack_tip_1_n,
                   const Eigen::Vector2d &crack_tip_2_t, const Eigen::Vector2d &crack_tip_2_n,
                   const Eigen::Matrix3d& D);