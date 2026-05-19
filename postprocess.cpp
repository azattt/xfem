#include "postprocess.h"

#include "gui.h"
#include "misc.h"

void drawHeavisideElements(const std::vector<HeavisideEnriched> &heaviside_enriched,
                           const QuadMesh &mesh,
                           const std::vector<HeavisideTriangulation> &heaviside_enriched_triangulation,
                           const std::vector<LevelSetSign> &vertices_level_set_signs,
                           const Eigen::VectorXd &u,
                           const std::vector<unsigned int> &node_offset,
                           const std::vector<Eigen::Vector2d> &vertices_displaced)
{
    constexpr double EPS = 1e-12;

    auto regularizeSign = [](int s) -> int {
        if (s > 0)
            return 1;
        if (s < 0)
            return -1;
        return 0;
    };

    for (int el = 0; el < heaviside_enriched.size(); ++el)
    {
        const HeavisideEnriched &hvsd_enr = heaviside_enriched[el];

        const std::array<int, 4> &element = mesh.elements[hvsd_enr.id];

        const HeavisideTriangulation &hvsd_trng =
            heaviside_enriched_triangulation[el];

        // ------------------------------------------------------------
        // 1. Node signs (handle zero-sign nodes robustly)
        // ------------------------------------------------------------
        std::array<int, 4> sRaw;
        std::array<int, 4> s;

        for (int i = 0; i < 4; ++i)
            sRaw[i] = regularizeSign(
                vertices_level_set_signs[element[i]].sign);

        for (int i = 0; i < 4; ++i)
        {
            if (sRaw[i] != 0)
            {
                s[i] = sRaw[i];
                continue;
            }

            // Crack passes through node
            // Borrow sign from neighboring nonzero node

            int prev = sRaw[(i + 3) % 4];
            int next = sRaw[(i + 1) % 4];

            if (prev != 0)
                s[i] = prev;
            else if (next != 0)
                s[i] = next;
            else
                s[i] = 1; // pathological fallback
        }

        std::cout << "Element signs: ";
        for (int i = 0; i < 4; ++i)
            std::cout << sRaw[i] << " ";
        std::cout << std::endl;

        // ------------------------------------------------------------
        // 2. Enrichment DOFs
        // ------------------------------------------------------------
        std::array<Eigen::Vector2d, 4> a;

        for (int i = 0; i < 4; ++i)
        {
            int off = node_offset[element[i]];

            a[i] = Eigen::Vector2d(
                u(off + 2),
                u(off + 3));
        }

        // ------------------------------------------------------------
        // 3. Standard displaced corner positions
        // ------------------------------------------------------------
        std::array<Eigen::Vector2d, 4> corners;

        for (int i = 0; i < 4; ++i)
            corners[i] = vertices_displaced[element[i]];

        // ------------------------------------------------------------
        // 4. Bilinear shape functions
        // ------------------------------------------------------------
        auto N = [](double xi, double eta)
            -> std::array<double, 4>
        {
            return {
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta)
            };
        };

        // ------------------------------------------------------------
        // 5. XFEM enriched position
        // ------------------------------------------------------------
        auto enrichedPos =
            [&](double xi, double eta, int Hside)
            -> Eigen::Vector2d
        {
            auto Ni = N(xi, eta);

            Eigen::Vector2d base =
                Eigen::Vector2d::Zero();

            for (int i = 0; i < 4; ++i)
                base += Ni[i] * corners[i];

            Eigen::Vector2d enrich =
                Eigen::Vector2d::Zero();

            for (int i = 0; i < 4; ++i)
            {
                // shifted Heaviside enrichment
                double shift =
                    static_cast<double>(Hside - s[i]);

                enrich += Ni[i] * shift * a[i];
            }

            return base + enrich;
        };

        // ------------------------------------------------------------
        // 6. Crack intersection points
        // ------------------------------------------------------------
        Eigen::Vector2d interPos[2];
        Eigen::Vector2d interNeg[2];

        for (int k = 0; k < 2; ++k)
        {
            double xi =
                hvsd_enr.intersection_points_local_coords[k].x();

            double eta =
                hvsd_enr.intersection_points_local_coords[k].y();

            interPos[k] = enrichedPos(xi, eta, +1);
            interNeg[k] = enrichedPos(xi, eta, -1);

            // If crack passes through node,
            // positive/negative points can become identical.
            // Collapse them to avoid tiny sliver triangles.

            if ((interPos[k] - interNeg[k]).norm() < EPS)
            {
                Eigen::Vector2d avg =
                    0.5 * (interPos[k] + interNeg[k]);

                interPos[k] = avg;
                interNeg[k] = avg;
            }

            std::cout << "Intersection " << k << "\n";
            std::cout << "  POS: "
                      << interPos[k].transpose() << "\n";
            std::cout << "  NEG: "
                      << interNeg[k].transpose() << "\n";
        }

        // ------------------------------------------------------------
        // 7. Build point arrays
        // ------------------------------------------------------------
        std::array<Eigen::Vector2d, 6> pointsPos;
        std::array<Eigen::Vector2d, 6> pointsNeg;

        for (int i = 0; i < 4; ++i)
        {
            pointsPos[i] = corners[i];
            pointsNeg[i] = corners[i];
        }

        pointsPos[4] = interPos[0];
        pointsPos[5] = interPos[1];

        pointsNeg[4] = interNeg[0];
        pointsNeg[5] = interNeg[1];

        // ------------------------------------------------------------
        // 8. Triangle area check
        // ------------------------------------------------------------
        auto triangleArea =
            [](const Eigen::Vector2d &a,
               const Eigen::Vector2d &b,
               const Eigen::Vector2d &c)
        {
            return 0.5 *
                   std::abs(
                       (b.x() - a.x()) * (c.y() - a.y()) -
                       (b.y() - a.y()) * (c.x() - a.x()));
        };

        // ------------------------------------------------------------
        // 9. Draw positive side
        // ------------------------------------------------------------
        for (unsigned int i = 0;
             i < hvsd_trng.positive_heaviside_triangles_num;
             ++i)
        {
            const auto &tri = hvsd_trng.tri_indices[i];

            const Eigen::Vector2d &p0 =
                pointsPos[tri[0]];

            const Eigen::Vector2d &p1 =
                pointsPos[tri[1]];

            const Eigen::Vector2d &p2 =
                pointsPos[tri[2]];

            if (triangleArea(p0, p1, p2) < EPS)
                continue;

            TriangleGUI::Renderer::instance().addTriangle(
                TriangleGUI::TriangleColored{
                    toGlm(p0.cast<float>().eval()),
                    toGlm(p1.cast<float>().eval()),
                    toGlm(p2.cast<float>().eval()),
                    TriangleGUI::packColor(
                        glm::vec4(0.f, 1.f, 0.f, 1.f))
                });
        }

        
        // ------------------------------------------------------------
        // 10. Draw negative side
        // ------------------------------------------------------------
        for (unsigned int i =
                 hvsd_trng.positive_heaviside_triangles_num;
             i < hvsd_trng.tri_indices.size();
             ++i)
        {
            const auto &tri = hvsd_trng.tri_indices[i];

            const Eigen::Vector2d &p0 =
                pointsNeg[tri[0]];

            const Eigen::Vector2d &p1 =
                pointsNeg[tri[1]];

            const Eigen::Vector2d &p2 =
                pointsNeg[tri[2]];

            if (triangleArea(p0, p1, p2) < EPS)
                continue;

            TriangleGUI::Renderer::instance().addTriangle(
                TriangleGUI::TriangleColored{
                    toGlm(p0.cast<float>().eval()),
                    toGlm(p1.cast<float>().eval()),
                    toGlm(p2.cast<float>().eval()),
                    TriangleGUI::packColor(
                        glm::vec4(0.f, 0.f, 1.f, 1.f))
                });
        }
    }
}