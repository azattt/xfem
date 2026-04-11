#include "levelset.h"

std::vector<Enriched> find_enriched_elements(const QuadMesh& quad_mesh, const Crack& crack){
    std::vector<unsigned int> triangle_indices;
    triangle_indices.reserve(quad_mesh.elements.size()*6);
    for (const QuadElement& quad: quad_mesh.elements){
        triangle_indices.insert(triangle_indices.end(), {
            quad.v0, quad.v1, quad.v2,
            quad.v0, quad.v2, quad.v3
        });
    }
    tinybvh::BVH bvh;
    bvh.Build(quad_mesh.vertices.data(), triangle_indices.data(), quad_mesh.elements.size()*2);

    tinybvh::bvhvec3 O( 0, 0, 0 );
	tinybvh::bvhvec3 D( 1.0f, 0, 0 );
    tinybvh::Ray ray;
    float px = -66.0f, py = -66.0f;
    ray.O = {px, py, 0.0f};
    ray.D = {-1.0f, 0.0f, 0.0f};
    
    if (bvh.Intersect(ray)) {
        // An intersection was found.
        std::cout << "Point (" << px << ", " << py << ") is inside the mesh! " << ray.hit.t << std::endl;
    } else {
        std::cout << "Point (" << px << ", " << py << ") is outside the mesh." << std::endl;
    }
    return std::vector<Enriched>();
}