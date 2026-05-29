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
    for (int i = 0; i < heaviside_enriched.size(); i++){
        const HeavisideEnriched& hvsd_enr = heaviside_enriched[i];
        const std::array<int, 4>& element = mesh.elements[hvsd_enr.id];
        const HeavisideTriangulation& hvsd_trng = heaviside_enriched_triangulation[i];
        const Eigen::Vector4d u = 
        for (int j = 0; j < hvsd_trng.positive_heaviside_triangles_num; j++){
            for (int k = 0; k < 3; k++){
                if (hvsd_trng.tri_indices[j][k] > 0){
                    vertices_displaced[element[0]]
                    glm::vec2 v0 = N1[0]*u0+N2[1]*u1+N3[2]*u2+N4[3]*u3
                    +2*N[0]*a0+2*N[1]*a1+2*N[2]*a2+2*N[3]*a3;
                    glm::vec2 v1 = N[0]*u0+N[1]*u1+N[2]*u2+N[3]*u3
                    +2*N[0]*a0+2*N[1]*a1+2*N[2]*a2+2*N[3]*a3;
                    glm::vec2 v2 = N[0]*u0+N[1]*u1+N[2]*u2+N[3]*u3
                    +2*N[0]*a0+2*N[1]*a1+2*N[2]*a2+2*N[3]*a3;
                    TriangleGUI::Renderer::instance().addTriangle(
                        TriangleGUI::TriangleColored{v0, v1, v2,
                            TriangleGUI::packColor(glm::vec4{0.0, 1.0, 0.0, 1.0})
                        }
                    );
                }
            }
        }
        for (int j = 0; j < hvsd_trng.positive_heaviside_triangles_num; j++){
            for (int k = 0; k < 3; k++){
                if (hvsd_trng.tri_indices[j][k] > 0){

                }
            }
        }
    }
}