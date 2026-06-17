#include "postprocess.h"

#include <algorithm>
#include <limits>

#include "fem.h"
#include "misc.h"

constexpr double PI = 3.14159265358979323846;
constexpr double INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2π)

void drawHeavisideElements(
    const std::vector<HeavisideEnriched>& heaviside_enriched,
    const QuadMesh& mesh,
    const std::vector<HeavisideTriangulation>& heaviside_enriched_triangulation,
    const std::vector<LevelSetSign>& vertices_level_set_signs,
    const Eigen::VectorXd& u_solu,
    const std::vector<unsigned int>& node_offset,
    const std::vector<bool>& heaviside_enriched_nodes,
    double scale
)
{
    auto quadShape = [](double xi, double eta) -> Eigen::RowVector4d {
        return Eigen::RowVector4d{
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta)
        };
    };

    for (int h_id = 0; h_id < static_cast<int>(heaviside_enriched.size()); ++h_id)
    {
        const HeavisideEnriched& hvsd_enr = heaviside_enriched[h_id];
        const std::array<int, 4>& element = mesh.elements[hvsd_enr.id];
        const HeavisideTriangulation& hvsd_trng =
            heaviside_enriched_triangulation[h_id];

        std::array<Eigen::Vector2d, 6> local_points;
        local_points[0] = Eigen::Vector2d{-1.0, -1.0};
        local_points[1] = Eigen::Vector2d{ 1.0, -1.0};
        local_points[2] = Eigen::Vector2d{ 1.0,  1.0};
        local_points[3] = Eigen::Vector2d{-1.0,  1.0};
        local_points[4] = hvsd_enr.intersection_points_local_coords[0];
        local_points[5] = hvsd_enr.intersection_points_local_coords[1];

        Eigen::Matrix<double, 4, 2> vertices;
        Eigen::Matrix<double, 4, 2> u_std;
        Eigen::Matrix<double, 4, 2> a_h;

        for (int n = 0; n < 4; ++n)
        {
            const int node = element[n];
            const int off = node_offset[node];

            vertices.row(n) = mesh.vertices[node];

            u_std.row(n) =
                u_solu.segment<2>(off).transpose() * scale;

            if (heaviside_enriched_nodes[node])
            {
                a_h.row(n) =
                    u_solu.segment<2>(off + 2).transpose() * scale;
            }
            else
            {
                a_h.row(n).setZero();
            }
        }

        auto evalHeavisidePoint = [&](
            double xi,
            double eta,
            int H_value
        ) -> Eigen::Vector2d {
            const Eigen::RowVector4d N = quadShape(xi, eta);

            Eigen::Vector2d p =
                (N * vertices).transpose();

            p += (N * u_std).transpose();

            for (int n = 0; n < 4; ++n)
            {
                const int node = element[n];

                const double H_i =
                    static_cast<double>(
                        vertices_level_set_signs[node].sign
                    );

                const double H_shift =
                    static_cast<double>(H_value) - H_i;

                p += N[n] * H_shift * a_h.row(n).transpose();
            }

            return p;
        };

        auto drawSide = [&](
            int tri_begin,
            int tri_end,
            int H_value,
            const glm::vec4& color
        ) {
            for (int tri_id = tri_begin; tri_id < tri_end; ++tri_id)
            {
                std::array<Eigen::Vector2d, 3> v;

                for (int k = 0; k < 3; ++k)
                {
                    const unsigned char local_id =
                        hvsd_trng.tri_indices[tri_id][k];

                    const double xi =
                        local_points[local_id].x();

                    const double eta =
                        local_points[local_id].y();

                    v[k] = evalHeavisidePoint(xi, eta, H_value);
                }

                TriangleGUI::Renderer::instance().addTriangle(
                    TriangleGUI::TriangleColored{
                        toGlm(v[0].cast<float>().eval()),
                        toGlm(v[1].cast<float>().eval()),
                        toGlm(v[2].cast<float>().eval()),
                        TriangleGUI::packColor(color)
                    }
                );
            }
        };

        // Верхняя/первая сторона
        drawSide(
            0,
            hvsd_trng.positive_heaviside_triangles_num,
            +1,
            glm::vec4{0.0, 1.0, 0.0, 1.0}
        );

        // Нижняя/вторая сторона
        drawSide(
            hvsd_trng.positive_heaviside_triangles_num,
            hvsd_trng.triangles_num,
            -1,
            glm::vec4{0.0, 0.0, 1.0, 1.0}
        );
    }
}

void drawTipElements(
    const std::vector<TipEnriched>& tip_enriched,
    const std::vector<HeavisideEnriched>& heaviside_enriched,
    const QuadMesh& mesh,
    const Eigen::VectorXd& u_solu,
    const std::vector<unsigned int>& node_offset,
    const std::vector<bool>& heaviside_enriched_nodes,
    const std::vector<bool>& tip_enriched_nodes,
    const std::vector<LevelSetSign>& vertices_level_set_signs,
    double scale,
    std::vector<PolygonalChain>& polygonal_chains,
    const Eigen::Vector2d& crack_tip_1_t,
    const Eigen::Vector2d& crack_tip_1_n,
    const Eigen::Vector2d& crack_tip_2_t,
    const Eigen::Vector2d& crack_tip_2_n
)
{
    auto quadShape = [](double xi, double eta) -> Eigen::RowVector4d {
        return Eigen::RowVector4d{
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta)
        };
    };

    auto elementSize = [](
        const Eigen::Matrix<double, 4, 2>& vertices
    ) -> double {
        return 0.25 *
            (
                (vertices.row(1) - vertices.row(0)).norm() +
                (vertices.row(2) - vertices.row(1)).norm() +
                (vertices.row(3) - vertices.row(2)).norm() +
                (vertices.row(0) - vertices.row(3)).norm()
            );
    };

    auto evalGlobalFromLocal = [&](
        const std::array<int, 4>& element,
        double xi,
        double eta
    ) -> Eigen::Vector2d {
        const Eigen::RowVector4d N = quadShape(xi, eta);

        Eigen::Vector2d p = Eigen::Vector2d::Zero();

        for (int n = 0; n < 4; ++n)
        {
            p += N[n] * mesh.vertices[element[n]];
        }

        return p;
    };

    auto evalHeavisideDisplacement = [&](
        const HeavisideEnriched& h_enr,
        double xi,
        double eta,
        int H_value
    ) -> Eigen::Vector2d {
        const std::array<int, 4>& element =
            mesh.elements[h_enr.id];

        const Eigen::RowVector4d N = quadShape(xi, eta);

        Eigen::Vector2d p = Eigen::Vector2d::Zero();

        for (int n = 0; n < 4; ++n)
        {
            const int node = element[n];
            const int off = node_offset[node];

            p += N[n] * mesh.vertices[node];

            p += N[n] *
                 u_solu.segment<2>(off) *
                 scale;

            if (heaviside_enriched_nodes[node])
            {
                const double H_i =
                    static_cast<double>(
                        vertices_level_set_signs[node].sign
                    );

                const double H_shift =
                    static_cast<double>(H_value) - H_i;

                p += N[n] *
                     H_shift *
                     u_solu.segment<2>(off + 2) *
                     scale;
            }
        }

        return p;
    };

    for (int tip_id = 0; tip_id < static_cast<int>(tip_enriched.size()); ++tip_id)
    {
        const TipEnriched& tip_enr = tip_enriched[tip_id];
        const std::array<int, 4>& element =
            mesh.elements[tip_enr.id];

        constexpr int N_points = 100;

        Eigen::Matrix<double, 4, 2> vertices;

        for (int n = 0; n < 4; ++n)
        {
            vertices.row(n) = mesh.vertices[element[n]];
        }

        const double xi_tip =
            tip_enr.tip_point_local_coords.x();

        const double eta_tip =
            tip_enr.tip_point_local_coords.y();

        const double xi_inter =
            tip_enr.intersection_point_local_coords.x();

        const double eta_inter =
            tip_enr.intersection_point_local_coords.y();

        const Eigen::Vector2d tip_point_global_coords =
            evalGlobalFromLocal(element, xi_tip, eta_tip);

        const Eigen::Vector2d intersection_point_global_coords =
            evalGlobalFromLocal(element, xi_inter, eta_inter);

        const Eigen::Vector2d crack_dir_local =
            tip_enr.tip_point_local_coords -
            tip_enr.intersection_point_local_coords;

        Eigen::Matrix<double, 4, 2> u_std;

        for (int n = 0; n < 4; ++n)
        {
            const int node = element[n];
            const int off = node_offset[node];

            u_std.row(n) =
                u_solu.segment<2>(off).transpose() * scale;
        }

        std::array<Eigen::Matrix<double, 4, 2>, 4> b_nodes;

        for (int n = 0; n < 4; ++n)
        {
            const int node = element[n];

            if (!tip_enriched_nodes[node])
            {
                throw std::runtime_error(
                    "drawTipElements: not tip enriched node"
                );
            }

            const int off = node_offset[node];
            const int offset_tip = 4;

            for (int a = 0; a < 4; ++a)
            {
                b_nodes[n](a, 0) =
                    u_solu(off + offset_tip + 2 * a) * scale;

                b_nodes[n](a, 1) =
                    u_solu(off + offset_tip + 2 * a + 1) * scale;
            }
        }

        const Eigen::Vector2d t_vec =
            (
                tip_enr.tip_index == 1
                    ? crack_tip_1_t
                    : crack_tip_2_t
            ).normalized();

        const Eigen::Vector2d n_vec =
            (
                tip_enr.tip_index == 1
                    ? crack_tip_1_n
                    : crack_tip_2_n
            ).normalized();

        std::array<std::array<double, 4>, 4> f_nodes;

        for (int n = 0; n < 4; ++n)
        {
            const Eigen::Vector2d d =
                vertices.row(n).transpose() -
                tip_point_global_coords;

            const double radius = d.norm();

            double theta = 0.0;

            if (radius > 1e-30)
            {
                theta =
                    std::atan2(
                        d.dot(n_vec),
                        d.dot(t_vec)
                    );
            }

            const double sqrt_r =
                std::sqrt(radius);

            const double sinhalftheta =
                std::sin(theta / 2.0);

            const double coshalftheta =
                std::cos(theta / 2.0);

            const double sintheta =
                std::sin(theta);

            f_nodes[n] = {
                sqrt_r * sinhalftheta,
                sqrt_r * coshalftheta,
                sqrt_r * sintheta * sinhalftheta,
                sqrt_r * sintheta * coshalftheta
            };
        }

        auto evalTipRawPoint = [&](
            double xi,
            double eta,
            int side
        ) -> Eigen::Vector2d {
            const Eigen::RowVector4d N =
                quadShape(xi, eta);

            const Eigen::Vector2d p =
                (N * vertices).transpose();

            const double r =
                (p - tip_point_global_coords).norm();

            const double sqrt_r =
                std::sqrt(r);

            // Твой вариант для crack faces:
            // upper: +1, 0, 0
            // lower: -1, 0, 0
            const double sinhalftheta =
                side > 0 ? 1.0 : -1.0;

            const double coshalftheta = 0.0;
            const double sintheta = 0.0;

            const std::array<double, 4> f = {
                sqrt_r * sinhalftheta,
                sqrt_r * coshalftheta,
                sqrt_r * sintheta * sinhalftheta,
                sqrt_r * sintheta * coshalftheta
            };

            Eigen::Vector2d p_def =
                p + (N * u_std).transpose();

            for (int n = 0; n < 4; ++n)
            {
                for (int a = 0; a < 4; ++a)
                {
                    const double shift =
                        f[a] - f_nodes[n][a];

                    p_def +=
                        N[n] *
                        shift *
                        b_nodes[n].row(a).transpose();
                }
            }

            return p_def;
        };

        bool found_heaviside_match = false;
        int best_h_id = -1;
        int best_h_intersection_id = -1;
        double best_dist = 1e100;

        const double h_tip =
            elementSize(vertices);

        for (int h_id = 0; h_id < static_cast<int>(heaviside_enriched.size()); ++h_id)
        {
            const HeavisideEnriched& h_enr =
                heaviside_enriched[h_id];

            const std::array<int, 4>& h_elem =
                mesh.elements[h_enr.id];

            for (int ip = 0; ip < 2; ++ip)
            {
                const double h_xi =
                    h_enr.intersection_points_local_coords[ip].x();

                const double h_eta =
                    h_enr.intersection_points_local_coords[ip].y();

                const Eigen::Vector2d h_p =
                    evalGlobalFromLocal(h_elem, h_xi, h_eta);

                const double dist =
                    (h_p - intersection_point_global_coords).norm();

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_h_id = h_id;
                    best_h_intersection_id = ip;
                }
            }
        }

        if (best_h_id >= 0 && best_dist < 0.25 * h_tip)
        {
            found_heaviside_match = true;
        }

        Eigen::Vector2d upper_correction =
            Eigen::Vector2d::Zero();

        Eigen::Vector2d lower_correction =
            Eigen::Vector2d::Zero();

        if (found_heaviside_match)
        {
            const HeavisideEnriched& h_enr =
                heaviside_enriched[best_h_id];

            const Eigen::Vector2d h_local =
                h_enr.intersection_points_local_coords
                    [best_h_intersection_id];

            const Eigen::Vector2d h_upper =
                evalHeavisideDisplacement(
                    h_enr,
                    h_local.x(),
                    h_local.y(),
                    +1
                );

            const Eigen::Vector2d h_lower =
                evalHeavisideDisplacement(
                    h_enr,
                    h_local.x(),
                    h_local.y(),
                    -1
                );

            const Eigen::Vector2d tip_upper_at_intersection =
                evalTipRawPoint(
                    xi_inter,
                    eta_inter,
                    +1
                );

            const Eigen::Vector2d tip_lower_at_intersection =
                evalTipRawPoint(
                    xi_inter,
                    eta_inter,
                    -1
                );

            upper_correction =
                h_upper - tip_upper_at_intersection;

            lower_correction =
                h_lower - tip_lower_at_intersection;
        }
        else
        {
            std::cout
                << "WARNING: drawTipElements: no adjacent Heaviside match. "
                << "tip_index = "
                << static_cast<int>(tip_enr.tip_index)
                << ", best_dist = "
                << best_dist
                << ", h_tip = "
                << h_tip
                << std::endl;
        }

        std::vector<glm::vec2> upper_points;
        std::vector<glm::vec2> lower_points;

        upper_points.reserve(N_points + 1);
        lower_points.reserve(N_points + 1);

        for (int ip = 0; ip <= N_points; ++ip)
        {
            const double alpha =
                static_cast<double>(ip) /
                static_cast<double>(N_points);

            const double xi =
                xi_inter +
                crack_dir_local.x() * alpha;

            const double eta =
                eta_inter +
                crack_dir_local.y() * alpha;

            Eigen::Vector2d p =
                evalTipRawPoint(xi, eta, +1);

            // alpha = 0: intersection, correction full
            // alpha = 1: tip, correction zero
            const double blend = 1.0 - alpha;

            p += blend * upper_correction;

            upper_points.push_back(
                glm::vec2{
                    static_cast<float>(p.x()),
                    static_cast<float>(p.y())
                }
            );
        }

        for (int ip = N_points; ip >= 0; --ip)
        {
            const double alpha =
                static_cast<double>(ip) /
                static_cast<double>(N_points);

            const double xi =
                xi_inter +
                crack_dir_local.x() * alpha;

            const double eta =
                eta_inter +
                crack_dir_local.y() * alpha;

            Eigen::Vector2d p =
                evalTipRawPoint(xi, eta, -1);

            const double blend = 1.0 - alpha;

            p += blend * lower_correction;

            lower_points.push_back(
                glm::vec2{
                    static_cast<float>(p.x()),
                    static_cast<float>(p.y())
                }
            );
        }

        PolygonalChain poly_chain_upper;
        poly_chain_upper.color =
            glm::vec4{1.0, 0.0, 1.0, 1.0};
        poly_chain_upper.points = upper_points;

        polygonal_chains.push_back(poly_chain_upper);

        PolygonalChain poly_chain_lower;
        poly_chain_lower.color =
            glm::vec4{1.0, 1.0, 0.0, 1.0};
        poly_chain_lower.points = lower_points;

        polygonal_chains.push_back(poly_chain_lower);
    }
}

double computeEquivalentK(double KI, double KII)
{
    return std::sqrt(KI * KI + KII * KII);
}

double computeCrackGrowthAngle(double KI, double KII)
{
    const double eps = 1e-14;

    if (std::abs(KII) < eps * std::max(1.0, std::abs(KI)))
    {
        return 0.0;
    }

    return 2.0 * std::atan(
        (KI - std::sqrt(KI * KI + 8.0 * KII * KII)) /
        (4.0 * KII)
    );
}
Eigen::Vector2d rotateVector(const Eigen::Vector2d& v, double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    return Eigen::Vector2d{
        c * v.x() - s * v.y(),
        s * v.x() + c * v.y()
    };
}
bool pointInsideDomain(
    const Eigen::Vector2d& p,
    double w,
    double h
)
{
    return p.x() >= 0.0 && p.x() <= w &&
           p.y() >= 0.0 && p.y() <= h;
}
bool growCrackOneStep(
    Crack& crack,
    const std::vector<TipKResult>& k_results,
    double KIC,
    double da,
    double domain_w,
    double domain_h
)
{
    if (k_results.empty())
    {
        std::cout << "No K results. Crack will not grow.\n";
        return false;
    }

    const TipKResult* best = nullptr;
    double best_Keq = -1.0;

    for (const TipKResult& r : k_results)
    {
        const double Keq = computeEquivalentK(r.K_I, r.K_II);

        if (Keq > best_Keq)
        {
            best_Keq = Keq;
            best = &r;
        }
    }

    if (best == nullptr)
    {
        return false;
    }

    std::cout << "Growth check: tip = " << best->tip_index
              << ", KI = " << best->K_I
              << ", KII = " << best->K_II
              << ", Keq = " << best_Keq
              << ", KIC = " << KIC
              << std::endl;

    if (best_Keq < KIC)
    {
        std::cout << "Crack does not grow: Keq < KIC\n";
        return false;
    }

    if (crack.vertices.size() < 2 || crack.indices.empty())
    {
        throw std::runtime_error("Crack must contain at least one segment");
    }

    Eigen::Vector2d old_tip;
    Eigen::Vector2d base_dir;

    if (best->tip_index == 1)
    {
        const CrackSegment& first_segment = crack.indices.front();

        old_tip = crack.vertices[first_segment.v0];

        // Для tip 1 направление наружу: от второго узла к первому.
        base_dir =
            (crack.vertices[first_segment.v0] -
             crack.vertices[first_segment.v1]).normalized();
    }
    else if (best->tip_index == 2)
    {
        const CrackSegment& last_segment = crack.indices.back();

        old_tip = crack.vertices[last_segment.v1];

        // Для tip 2 направление наружу: от предпоследней точки к последней.
        base_dir =
            (crack.vertices[last_segment.v1] -
             crack.vertices[last_segment.v0]).normalized();
    }
    else
    {
        throw std::runtime_error("Invalid tip_index in growCrackOneStep");
    }

    const double theta = computeCrackGrowthAngle(best->K_I, best->K_II);

    Eigen::Vector2d growth_dir = rotateVector(base_dir, theta).normalized();

    Eigen::Vector2d new_tip = old_tip + da * growth_dir;

    // if (!pointInsideDomain(new_tip, domain_w, domain_h))
    // {
    //     std::cout << "Crack growth stopped: new tip is outside domain. "
    //               << "new_tip = " << new_tip.transpose() << "\n";
    //     return false;
    // }

    if (best->tip_index == 1)
    {
        const int old_first_index = crack.indices.front().v0;

        const int new_index =
            static_cast<int>(crack.vertices.size());

        crack.vertices.push_back(new_tip);

        // Новый сегмент: new_tip -> old_tip.
        // Он становится первым сегментом трещины.
        crack.indices.insert(
            crack.indices.begin(),
            CrackSegment{new_index, old_first_index}
        );
    }
    else
    {
        const int old_last_index = crack.indices.back().v1;

        const int new_index =
            static_cast<int>(crack.vertices.size());

        crack.vertices.push_back(new_tip);

        // Новый сегмент: old_tip -> new_tip.
        crack.indices.push_back(
            CrackSegment{old_last_index, new_index}
        );
    }

    std::cout << "Crack grown at tip " << best->tip_index
              << ", theta = " << theta
              << ", old_tip = " << old_tip.transpose()
              << ", new_tip = " << new_tip.transpose()
              << std::endl;

    return true;
}

template <unsigned int NGauss>
std::vector<TipKResult> computeStress(const std::vector<TipEnriched> &tip_enriched,
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
    std::vector<TipKResult> results;

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
                                    double xi, double eta, double integration_weight, int forced_H_value,
                                    bool use_heaviside_in_this_element, bool use_tip_in_this_element) {
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

                if (use_heaviside_in_this_element && heaviside_enriched_nodes[node])
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

                if (use_tip_in_this_element && tip_enriched_nodes[node])
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

        // ------------------------------------------------------------
        // Duffy quadrature on [0, 1] x [0, 1]
        // Used only for tip subtriangles.
        // ------------------------------------------------------------
        constexpr int DuffyN = 8;

        const std::array<double, DuffyN> duffy_pts = {0.0198550717512319, 0.101666761293187, 0.237233795041836,
                                                      0.408282678752175,  0.591717321247825, 0.762766204958164,
                                                      0.898333238706813,  0.980144928248768};

        const std::array<double, DuffyN> duffy_wts = {0.0506142681451881, 0.111190517226687, 0.156853322938944,
                                                      0.181341891689181,  0.181341891689181, 0.156853322938944,
                                                      0.111190517226687,  0.0506142681451881};

        auto det2 = [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) -> double {
            return a.x() * b.y() - a.y() * b.x();
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
            // 1. Current crack-tip element: use Duffy transform
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

                    // ------------------------------------------------------------
                    // Duffy transform must start from the singular vertex.
                    // In this local_points array:
                    //
                    // local_points[5] = crack tip
                    //
                    // Therefore every tip subtriangle must contain vertex 5.
                    // ------------------------------------------------------------
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
                            "Duffy transform failed: tip point is not a vertex of tip subtriangle");
                    }

                    const int idx_A = triangle[tip_pos]; // tip vertex
                    const int idx_B = triangle[(tip_pos + 1) % 3];
                    const int idx_C = triangle[(tip_pos + 2) % 3];

                    const Eigen::Vector2d A = local_points[idx_A]; // tip point
                    const Eigen::Vector2d Bp = local_points[idx_B];
                    const Eigen::Vector2d Cp = local_points[idx_C];

                    const Eigen::Vector2d AB = Bp - A;
                    const Eigen::Vector2d AC = Cp - A;

                    const double det_duffy_tri = std::abs(det2(AB, AC));

                    if (det_duffy_tri < 1e-14)
                    {
                        throw std::runtime_error("Degenerate Duffy subtriangle");
                    }

                    for (int ir = 0; ir < DuffyN; ++ir)
                    {
                        const double rho = duffy_pts[ir];
                        const double wrho = duffy_wts[ir];

                        for (int it = 0; it < DuffyN; ++it)
                        {
                            const double tau = duffy_pts[it];
                            const double wtau = duffy_wts[it];

                            // ------------------------------------------------------------
                            // Duffy mapping from square to triangle:
                            //
                            // X(rho,tau) =
                            // A + rho * ((1 - tau) * (B - A) + tau * (C - A))
                            //
                            // rho = 0 -> crack tip
                            // rho = 1 -> opposite edge B-C
                            //
                            // Jacobian in local xi-eta triangle:
                            //
                            // |d(xi,eta)/d(rho,tau)| =
                            // rho * |det(B-A, C-A)|
                            // ------------------------------------------------------------
                            const Eigen::Vector2d xi_eta = A + rho * ((1.0 - tau) * AB + tau * AC);

                            const double xi = xi_eta.x();
                            const double eta = xi_eta.y();

                            LinearQuad::ShapeData tmp_shape = fillQuadShape(xi, eta);

                            LinearTriangle::JacobianData tmp_jd;
                            tmp_jd.J = tmp_shape.dN_xi_eta * coords;

                            bool tmp_invertible = false;

                            tmp_jd.J.computeInverseAndDetWithCheck(tmp_jd.invJ, tmp_jd.detJ, tmp_invertible, 1e-12);

                            if (!tmp_invertible)
                            {
                                throw std::runtime_error("Jacobi matrix is not invertible");
                            }

                            const double weight = wrho * wtau * rho * det_duffy_tri * std::abs(tmp_jd.detJ);

                            integrateAtPoint(element, coords, xi, eta, weight, 0,
                                             false, // Heaviside не включаем в tip element
                                             true   // tip enrichment включаем
                            );
                        }
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

                    const int forced_H_value = tri_id < h_triangulation.positive_heaviside_triangles_num ? -1 : 1;

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

                        integrateAtPoint(element, coords, xi, eta, weight, forced_H_value,
                                         true, // Heaviside включаем
                                         false // tip не включаем
                        );
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

                        integrateAtPoint(element, coords, xi, eta, weight, 0,
                                         false, // Heaviside не включаем
                                         false  // tip не включаем
                        );
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
        results.push_back(TipKResult{
            static_cast<int>(tip_data.tip_index),
            K_I,
            K_II,
            used_elements
        });
    }
    return results;
}

// It turns out that for templates we need to either implement a function in a header file or explicitly create a
// template as shown below
template std::vector<TipKResult> computeStress<13>(
    const std::vector<TipEnriched> &tip_enriched, const std::vector<HeavisideEnriched> &heaviside_enriched,
    const QuadMesh &mesh, const std::vector<TipTriangulation> &tip_enriched_triangulation,
    const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation, const Eigen::VectorXd &u_solu,
    const std::vector<unsigned int> &node_offset, const std::vector<bool> &heaviside_enriched_nodes,
    const std::vector<bool> &tip_enriched_nodes, const std::vector<LevelSetSign> &vertices_level_set_signs,
    const std::array<std::array<double, 2>, 13> &gauss_pts, const std::array<double, 13> &gauss_wts,
    const Eigen::Vector2d &crack_tip_1_t, const Eigen::Vector2d &crack_tip_1_n, const Eigen::Vector2d &crack_tip_2_t,
    const Eigen::Vector2d &crack_tip_2_n, const Eigen::Matrix3d &D, const double young_modulus,
    const double poisson_ratio, const double Rin, const double Rout);


void drawVonMisesStressField(
    const std::vector<TipEnriched>& tip_enriched,
    const std::vector<HeavisideEnriched>& heaviside_enriched,
    const QuadMesh& mesh,
    const std::vector<TipTriangulation>& tip_enriched_triangulation,
    const std::vector<HeavisideTriangulation>& heaviside_enriched_triangulation,
    const Eigen::VectorXd& u_solu,
    const std::vector<unsigned int>& node_offset,
    const std::vector<bool>& heaviside_enriched_nodes,
    const std::vector<bool>& tip_enriched_nodes,
    const std::vector<LevelSetSign>& vertices_level_set_signs,
    const Eigen::Vector2d& crack_tip_1_t,
    const Eigen::Vector2d& crack_tip_1_n,
    const Eigen::Vector2d& crack_tip_2_t,
    const Eigen::Vector2d& crack_tip_2_n,
    const Eigen::Matrix3d& D,
    double scale
)
{
    struct StressTriangle
    {
        Eigen::Vector2d p0;
        Eigen::Vector2d p1;
        Eigen::Vector2d p2;
        double value = 0.0;
    };

    struct TipContext
    {
        bool valid = false;
        Eigen::Vector2d tip_point = Eigen::Vector2d::Zero();
        Eigen::Vector2d t_vec = Eigen::Vector2d::UnitX();
        Eigen::Vector2d n_vec = Eigen::Vector2d::UnitY();
        std::array<std::array<double, 4>, 4> f_nodes;
    };

    auto quadShape = [](double xi, double eta) {
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

    auto vonMisesPlaneStress = [](const Eigen::Vector3d& stress) -> double {
        const double sx = stress(0);
        const double sy = stress(1);
        const double txy = stress(2);

        return std::sqrt(
            sx * sx -
            sx * sy +
            sy * sy +
            3.0 * txy * txy
        );
    };

    auto colorMap = [](double value, double vmin, double vmax) -> glm::vec4 {
        double t = 0.0;

        if (vmax > vmin)
        {
            t = (value - vmin) / (vmax - vmin);
        }

        t = std::clamp(t, 0.0, 1.0);

        double r = 0.0;
        double g = 0.0;
        double b = 0.0;

        if (t < 0.25)
        {
            const double q = t / 0.25;
            r = 0.0;
            g = q;
            b = 1.0;
        }
        else if (t < 0.5)
        {
            const double q = (t - 0.25) / 0.25;
            r = 0.0;
            g = 1.0;
            b = 1.0 - q;
        }
        else if (t < 0.75)
        {
            const double q = (t - 0.5) / 0.25;
            r = q;
            g = 1.0;
            b = 0.0;
        }
        else
        {
            const double q = (t - 0.75) / 0.25;
            r = 1.0;
            g = 1.0 - q;
            b = 0.0;
        }

        return glm::vec4{
            static_cast<float>(r),
            static_cast<float>(g),
            static_cast<float>(b),
            1.0f
        };
    };

    auto makeTipContext = [&](
        const TipEnriched& tip_data,
        const std::array<int, 4>& element,
        const Eigen::Matrix<double, 4, 2>& coords
    ) -> TipContext {
        TipContext ctx;
        ctx.valid = true;

        const double xi_tip = tip_data.tip_point_local_coords.x();
        const double eta_tip = tip_data.tip_point_local_coords.y();

        const LinearQuad::ShapeData tip_shape =
            quadShape(xi_tip, eta_tip);

        ctx.tip_point.setZero();

        for (int n = 0; n < 4; ++n)
        {
            ctx.tip_point +=
                tip_shape.N[n] * coords.row(n).transpose();
        }

        if (tip_data.tip_index == 1)
        {
            ctx.t_vec = crack_tip_1_t.normalized();
            ctx.n_vec = crack_tip_1_n.normalized();
        }
        else
        {
            ctx.t_vec = crack_tip_2_t.normalized();
            ctx.n_vec = crack_tip_2_n.normalized();
        }

        for (int n = 0; n < 4; ++n)
        {
            const Eigen::Vector2d d =
                coords.row(n).transpose() - ctx.tip_point;

            const double x1 = d.dot(ctx.t_vec);
            const double x2 = d.dot(ctx.n_vec);

            const double r = std::sqrt(x1 * x1 + x2 * x2);

            if (r < 1e-30)
            {
                ctx.f_nodes[n] = {0.0, 0.0, 0.0, 0.0};
                continue;
            }

            const double theta = std::atan2(x2, x1);
            const double sqrt_r = std::sqrt(r);
            const double sin_half = std::sin(theta / 2.0);
            const double cos_half = std::cos(theta / 2.0);
            const double sin_theta = std::sin(theta);

            ctx.f_nodes[n] = {
                sqrt_r * sin_half,
                sqrt_r * cos_half,
                sqrt_r * sin_theta * sin_half,
                sqrt_r * sin_theta * cos_half
            };
        }

        return ctx;
    };

    struct FieldValue
    {
        Eigen::Vector2d deformed_point = Eigen::Vector2d::Zero();
        Eigen::Vector3d stress = Eigen::Vector3d::Zero();
        double von_mises = 0.0;
    };

    auto evalFieldAtPoint = [&](
        const std::array<int, 4>& element,
        const Eigen::Matrix<double, 4, 2>& coords,
        double xi,
        double eta,
        int forced_H_value,
        bool use_heaviside,
        bool use_tip,
        const TipContext* tip_ctx
    ) -> FieldValue {
        FieldValue out;

        const LinearQuad::ShapeData shape =
            quadShape(xi, eta);

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
                "drawVonMisesStressField: Jacobi matrix is not invertible"
            );
        }

        const Eigen::Matrix<double, 2, 4> dN_dx_dy =
            jd.invJ * shape.dN_xi_eta;

        Eigen::Vector2d x_global = Eigen::Vector2d::Zero();

        for (int n = 0; n < 4; ++n)
        {
            x_global += shape.N[n] * coords.row(n).transpose();
        }

        Eigen::Vector2d u_global = Eigen::Vector2d::Zero();
        Eigen::Matrix2d grad_u;
        grad_u.setZero();

        for (int n = 0; n < 4; ++n)
        {
            const int node = element[n];
            const int off = node_offset[node];

            const double Nn = shape.N[n];
            const double dNdx = dN_dx_dy(0, n);
            const double dNdy = dN_dx_dy(1, n);

            const double ux = u_solu(off);
            const double uy = u_solu(off + 1);

            u_global += Nn * Eigen::Vector2d{ux, uy};

            grad_u(0, 0) += dNdx * ux;
            grad_u(0, 1) += dNdy * ux;
            grad_u(1, 0) += dNdx * uy;
            grad_u(1, 1) += dNdy * uy;

            if (use_heaviside && heaviside_enriched_nodes[node])
            {
                const int H_i =
                    vertices_level_set_signs[node].sign;

                const double H_shift =
                    static_cast<double>(forced_H_value - H_i);

                const double ax = u_solu(off + 2);
                const double ay = u_solu(off + 3);

                u_global +=
                    Nn * H_shift * Eigen::Vector2d{ax, ay};

                grad_u(0, 0) += dNdx * H_shift * ax;
                grad_u(0, 1) += dNdy * H_shift * ax;
                grad_u(1, 0) += dNdx * H_shift * ay;
                grad_u(1, 1) += dNdy * H_shift * ay;
            }
        }

        if (use_tip && tip_ctx != nullptr && tip_ctx->valid)
        {
            const Eigen::Vector2d d =
                x_global - tip_ctx->tip_point;

            const double x1 = d.dot(tip_ctx->t_vec);
            const double x2 = d.dot(tip_ctx->n_vec);

            const double r2 = x1 * x1 + x2 * x2;
            const double r = std::sqrt(r2);

            if (r > 1e-14)
            {
                const double theta = std::atan2(x2, x1);

                const double sqrt_r = std::sqrt(r);
                const double inv_sqrt_r = 1.0 / sqrt_r;

                const double sin_half = std::sin(theta / 2.0);
                const double cos_half = std::cos(theta / 2.0);
                const double sin_theta = std::sin(theta);
                const double cos_theta = std::cos(theta);

                std::array<double, 4> f = {
                    sqrt_r * sin_half,
                    sqrt_r * cos_half,
                    sqrt_r * sin_theta * sin_half,
                    sqrt_r * sin_theta * cos_half
                };

                std::array<double, 4> df_dr;
                std::array<double, 4> df_dtheta;

                df_dr[0] = 0.5 * inv_sqrt_r * sin_half;
                df_dr[1] = 0.5 * inv_sqrt_r * cos_half;
                df_dr[2] = 0.5 * inv_sqrt_r * sin_half * sin_theta;
                df_dr[3] = 0.5 * inv_sqrt_r * cos_half * sin_theta;

                df_dtheta[0] =
                    sqrt_r * 0.5 * cos_half;

                df_dtheta[1] =
                    -sqrt_r * 0.5 * sin_half;

                df_dtheta[2] =
                    sqrt_r *
                    (
                        0.5 * cos_half * sin_theta +
                        sin_half * cos_theta
                    );

                df_dtheta[3] =
                    sqrt_r *
                    (
                        -0.5 * sin_half * sin_theta +
                        cos_half * cos_theta
                    );

                const double drdx = d.x() / r;
                const double drdy = d.y() / r;

                const double dtheta_dx =
                    (x1 * tip_ctx->n_vec.x() -
                     x2 * tip_ctx->t_vec.x()) / r2;

                const double dtheta_dy =
                    (x1 * tip_ctx->n_vec.y() -
                     x2 * tip_ctx->t_vec.y()) / r2;

                std::array<Eigen::Vector2d, 4> df_dx;

                for (int a = 0; a < 4; ++a)
                {
                    df_dx[a].x() =
                        df_dr[a] * drdx +
                        df_dtheta[a] * dtheta_dx;

                    df_dx[a].y() =
                        df_dr[a] * drdy +
                        df_dtheta[a] * dtheta_dy;
                }

                for (int n = 0; n < 4; ++n)
                {
                    const int node = element[n];
                    const int off = node_offset[node];

                    const double Nn = shape.N[n];
                    const double dNdx = dN_dx_dy(0, n);
                    const double dNdy = dN_dx_dy(1, n);

                    if (!tip_enriched_nodes[node])
                    {
                        continue;
                    }

                    for (int a = 0; a < 4; ++a)
                    {
                        const double bx =
                            u_solu(off + 4 + 2 * a);

                        const double by =
                            u_solu(off + 4 + 2 * a + 1);

                        const double shift =
                            f[a] - tip_ctx->f_nodes[n][a];

                        const double d_enr_dx =
                            dNdx * shift +
                            Nn * df_dx[a].x();

                        const double d_enr_dy =
                            dNdy * shift +
                            Nn * df_dx[a].y();

                        u_global +=
                            Nn * shift * Eigen::Vector2d{bx, by};

                        grad_u(0, 0) += d_enr_dx * bx;
                        grad_u(0, 1) += d_enr_dy * bx;
                        grad_u(1, 0) += d_enr_dx * by;
                        grad_u(1, 1) += d_enr_dy * by;
                    }
                }
            }
        }

        out.deformed_point =
            x_global + scale * u_global;

        Eigen::Vector3d strain;
        strain << grad_u(0, 0),
                  grad_u(1, 1),
                  grad_u(0, 1) + grad_u(1, 0);

        out.stress = D * strain;
        out.von_mises = vonMisesPlaneStress(out.stress);

        return out;
    };

    std::vector<int> tip_element_to_index(
        mesh.elements.size(),
        -1
    );

    for (int i = 0; i < static_cast<int>(tip_enriched.size()); ++i)
    {
        tip_element_to_index[tip_enriched[i].id] = i;
    }

    std::vector<int> heaviside_element_to_index(
        mesh.elements.size(),
        -1
    );

    for (int i = 0; i < static_cast<int>(heaviside_enriched.size()); ++i)
    {
        heaviside_element_to_index[heaviside_enriched[i].id] = i;
    }

    std::vector<StressTriangle> stress_triangles;

    auto addStressTriangle = [&](
        const std::array<int, 4>& element,
        const Eigen::Matrix<double, 4, 2>& coords,
        const std::array<Eigen::Vector2d, 3>& local_tri,
        int forced_H_value,
        bool use_heaviside,
        bool use_tip,
        const TipContext* tip_ctx
    ) {
        const double xi_c =
            (local_tri[0].x() + local_tri[1].x() + local_tri[2].x()) / 3.0;

        const double eta_c =
            (local_tri[0].y() + local_tri[1].y() + local_tri[2].y()) / 3.0;

        const FieldValue c =
            evalFieldAtPoint(
                element,
                coords,
                xi_c,
                eta_c,
                forced_H_value,
                use_heaviside,
                use_tip,
                tip_ctx
            );

        const FieldValue v0 =
            evalFieldAtPoint(
                element,
                coords,
                local_tri[0].x(),
                local_tri[0].y(),
                forced_H_value,
                use_heaviside,
                use_tip,
                tip_ctx
            );

        const FieldValue v1 =
            evalFieldAtPoint(
                element,
                coords,
                local_tri[1].x(),
                local_tri[1].y(),
                forced_H_value,
                use_heaviside,
                use_tip,
                tip_ctx
            );

        const FieldValue v2 =
            evalFieldAtPoint(
                element,
                coords,
                local_tri[2].x(),
                local_tri[2].y(),
                forced_H_value,
                use_heaviside,
                use_tip,
                tip_ctx
            );

        StressTriangle tri;
        tri.p0 = v0.deformed_point;
        tri.p1 = v1.deformed_point;
        tri.p2 = v2.deformed_point;
        tri.value = c.von_mises;

        stress_triangles.push_back(tri);
    };

    for (int elem_id = 0; elem_id < static_cast<int>(mesh.elements.size()); ++elem_id)
    {
        const std::array<int, 4>& element =
            mesh.elements[elem_id];

        Eigen::Matrix<double, 4, 2> coords;

        for (int n = 0; n < 4; ++n)
        {
            coords.row(n) = mesh.vertices[element[n]];
        }

        const int tip_index =
            tip_element_to_index[elem_id];

        const int h_index =
            heaviside_element_to_index[elem_id];

        if (tip_index >= 0)
        {
            const TipEnriched& tip_data =
                tip_enriched[tip_index];

            const TipTriangulation& triangulation =
                tip_enriched_triangulation[tip_index];

            TipContext tip_ctx =
                makeTipContext(
                    tip_data,
                    element,
                    coords
                );

            std::array<Eigen::Vector2d, 6> local_points = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0},
                tip_data.intersection_point_local_coords,
                tip_data.tip_point_local_coords
            };

            for (unsigned int tri_id = 0; tri_id < 5; ++tri_id)
            {
                const std::array<unsigned char, 3>& tri =
                    triangulation.tri_indices[tri_id];

                std::array<Eigen::Vector2d, 3> local_tri = {
                    local_points[tri[0]],
                    local_points[tri[1]],
                    local_points[tri[2]]
                };

                addStressTriangle(
                    element,
                    coords,
                    local_tri,
                    0,
                    false,
                    true,
                    &tip_ctx
                );
            }
        }
        else if (h_index >= 0)
        {
            const HeavisideEnriched& h_data =
                heaviside_enriched[h_index];

            const HeavisideTriangulation& triangulation =
                heaviside_enriched_triangulation[h_index];

            std::array<Eigen::Vector2d, 6> local_points = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0},
                h_data.intersection_points_local_coords[0],
                h_data.intersection_points_local_coords[1]
            };

            for (unsigned int tri_id = 0;
                 tri_id < triangulation.triangles_num;
                 ++tri_id)
            {
                const std::array<unsigned char, 3>& tri =
                    triangulation.tri_indices[tri_id];

                std::array<Eigen::Vector2d, 3> local_tri = {
                    local_points[tri[0]],
                    local_points[tri[1]],
                    local_points[tri[2]]
                };

                // Это должно совпадать с твоей сборкой Heaviside.
                // В твоём main сейчас первая группа sign=+1,
                // после positive_heaviside_triangles_num sign=-1.
                const int H_value =
                    tri_id < triangulation.positive_heaviside_triangles_num
                        ? +1
                        : -1;

                addStressTriangle(
                    element,
                    coords,
                    local_tri,
                    H_value,
                    true,
                    false,
                    nullptr
                );
            }
        }
        else
        {
            const std::array<Eigen::Vector2d, 3> tri_1 = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0}
            };

            const std::array<Eigen::Vector2d, 3> tri_2 = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0}
            };

            addStressTriangle(
                element,
                coords,
                tri_1,
                0,
                false,
                false,
                nullptr
            );

            addStressTriangle(
                element,
                coords,
                tri_2,
                0,
                false,
                false,
                nullptr
            );
        }
    }

    if (stress_triangles.empty())
    {
        std::cout << "drawVonMisesStressField: no stress triangles\n";
        return;
    }

    double vmin = std::numeric_limits<double>::max();
    double vmax = -std::numeric_limits<double>::max();

    for (const StressTriangle& tri : stress_triangles)
    {
        if (!std::isfinite(tri.value))
        {
            continue;
        }

        vmin = std::min(vmin, tri.value);
        vmax = std::max(vmax, tri.value);
    }

    if (!(vmax > vmin))
    {
        vmax = vmin + 1.0;
    }

    std::cout << "Von Mises stress range: "
              << vmin << " ... " << vmax << std::endl;

    for (const StressTriangle& tri : stress_triangles)
    {
        const glm::vec4 color =
            colorMap(tri.value, vmin, vmax);

        TriangleGUI::Renderer::instance().addTriangle(
            TriangleGUI::TriangleColored{
                toGlm(tri.p0.cast<float>().eval()),
                toGlm(tri.p1.cast<float>().eval()),
                toGlm(tri.p2.cast<float>().eval()),
                TriangleGUI::packColor(color)
            }
        );
    }
}