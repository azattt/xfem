#include "levelset.h"

#include <cmath>
#include <limits>
#include <numeric> // std::iota

#include "gui.h"

// using namespace fcpw;

// // Helper: convert quad mesh to line segments (each quad → 4 edges)
// void convertQuadsToLineSegments(
//     const std::vector<Eigen::Vector<double, 2>>& vertices,
//     const std::vector<std::array<int, 4>>& quads,
//     std::vector<Eigen::Vector<double, 2>>& outPositions,      // flattened vertex list
//     std::vector<Vector2i>& outIndices) {       // edge indices
//     outPositions = vertices; // keep original vertices
//     for (const auto& q : quads) {
//         // add 4 edges per quad (counter-clockwise)
//         outIndices.push_back(Vector2i(q[0], q[1]));
//         outIndices.push_back(Vector2i(q[1], q[2]));
//         outIndices.push_back(Vector2i(q[2], q[3]));
//         outIndices.push_back(Vector2i(q[3], q[0]));
//     }
// }

void find_enriched_elements(const QuadMesh& quad_mesh, const Crack& crack){
    /*
    const unsigned int nTris = quad_mesh.elements.size()*2;
    
    std::vector<fcpw::Vector3i> triangle_indices;
    triangle_indices.reserve(quad_mesh.elements.size()*6);
    for (const std::array<int, 4>& quad: quad_mesh.elements){
        // bboxes.push_back(BB)
        triangle_indices.push_back(fcpw::Vector3i(quad[0], quad[1], quad[2]));
        triangle_indices.push_back(fcpw::Vector3i(quad[0], quad[2], quad[3]));
    }
    

//  ---- 2. Convert quads to line segments ----
    std::vector<Eigen::Vector<double, 2>> lineVertices;
    std::vector<Vector2i> lineIndices;
    convertQuadsToLineSegments(quad_mesh.vertices, quad_mesh.elements, lineVertices, lineIndices);

    // ---- 3. Setup FCPW scene and build BVH ----
    Scene<2> scene;
    scene.setObjectCount(1);
    scene.setObjectVertices(lineVertices, 0);
    scene.setObjectLineSegments(lineIndices, 0);
    // scene.setObjectVertices(quad_mesh.vertices, 0);
    // scene.setObjectTriangles(triangle_indices, 0);
    // scene.setObjectLineSegments(lineIndices, 0);
    // std::vector<Vector<3>> vertices = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}};
    // std::vector<Vector3i> indices = {{0, 1, 2}};
    // scene.setObjectVertices(vertices, 0);
    // scene.setObjectTriangles(indices, 0);
    AggregateType aggType = AggregateType::Bvh_SurfaceArea;
    bool vectorized = true;               // use Enoki vectorization
    bool printStats = true;
    bool reduceMemory = false;
    scene.build(aggType, vectorized, printStats, reduceMemory);

    // ---- 4. Define polyline (example: crossing the square) ----
    std::vector<Eigen::Vector<double, 2>> polyline = {
        Eigen::Vector<double, 2>(0.001f, 0.5f),  // start left of square
        Eigen::Vector<double, 2>(0.5f, 0.5f)   // end right of square
    };

    // ---- 5. Intersect each polyline segment ----
    bool anyHit = false;
    const float maxDist = 1000.0f;

    for (size_t i = 0; i + 1 < polyline.size(); ++i) {
        const Eigen::Vector<double, 2>& p0 = polyline[i];
        const Eigen::Vector<double, 2>& p1 = polyline[i + 1];
        Eigen::Vector<double, 2> dir = p1 - p0;
        const float segLen = dir.norm();
        if (segLen == 0.0f) continue;
        dir = dir / segLen;

        // Create ray
        Ray<2> ray(p0, dir, segLen);

        int nHit;
        // Perform intersection
        std::vector<Interaction<2>> interactions;
        nHit = scene.intersect(ray, interactions, false, true); // returns true if any hit

        if (nHit) {
            anyHit = true;
            for (const auto& inter : interactions) {
                int quadIndex = inter.primitiveIndex;
                float t = inter.d;
                Eigen::Vector<double, 2> point = inter.p;
                std::cout << "  Hit at t=" << t << ", quad=" << quadIndex  / 4
                        << " at (" << point[0] << "," << point[1] << ")\n";
            }
        }
    }

    if (!anyHit)
        std::cout << "No intersections found.\n";

    // Interaction<2> interaction;
    // scene.findClosestPoint(fcpw::Vector3(0.8f, 0.8f), interaction);
    
    return std::vector<HeavisideEnriched>();
    */
}



LevelSetFields compute_level_set_fields(const QuadMesh& quad_mesh, const Crack& crack){
    std::vector<LevelSetSign> vertices_level_set_signs;
    std::vector<double> level_set_1_signed_dist;
    std::vector<double> level_set_2_signed_dist;
    std::vector<double> level_set_3_signed_dist;
    std::vector<int> level_set_tips;
    const int total_vertices = quad_mesh.vertices.size();
    vertices_level_set_signs.resize(total_vertices);
    level_set_1_signed_dist.resize(total_vertices);
    level_set_2_signed_dist.resize(total_vertices);
    level_set_3_signed_dist.resize(total_vertices);
    level_set_tips.resize(total_vertices);

    if (crack.indices.empty()){
        return LevelSetFields{
            vertices_level_set_signs,
            level_set_1_signed_dist,
            level_set_2_signed_dist,
            level_set_3_signed_dist
        };
    }

    std::vector<CrackSegmentPrecomputed> line_segments;
    
    for (const CrackSegment &line : crack.indices)
    {
        const Eigen::Vector<double, 2> segment = crack.vertices[line.v1] - crack.vertices[line.v0];
        line_segments.push_back(CrackSegmentPrecomputed{crack.vertices[line.v0], segment, segment.dot(segment)});
    }
    const CrackSegmentPrecomputed& last_segment = line_segments[line_segments.size() - 1];
    const Eigen::Vector<double, 2> last_vertex = last_segment.v0 + last_segment.dir;
    const CrackSegmentPrecomputed& first_segment = line_segments[0];
    const Eigen::Vector<double, 2>& first_vertex = first_segment.v0;
    double line_segment_min_dist2 = 0;
    std::cout << "Setting level set values for nodes" << std::endl;
    for (size_t i = 0; i < quad_mesh.vertices.size(); i++)
    {
        // break;
        const Eigen::Vector<double, 2> &vertex = quad_mesh.vertices[i];
        line_segment_min_dist2 = std::numeric_limits<double>::max();
        size_t line_segment_min_dist_index = 0;
        double line_segment_min_dist_t_par = 0;
        for (size_t j = 0; j < line_segments.size(); j++)
        {
            const CrackSegmentPrecomputed &line_segment = line_segments[j];
            const Eigen::Vector<double, 2> v0_to_vertex = vertex - line_segment.v0;
            if (line_segment.l_squared < 1e-12)
            {
                std::cout << "Degenerate segment" << std::endl;
                continue;
            }
            
            const double t = v0_to_vertex.dot(line_segment.dir) / line_segment.l_squared;
            const Eigen::Vector<double, 2> closest_point = line_segment.v0 + std::clamp(t, 0.0, 1.0) * line_segment.dir;
            const Eigen::Vector<double, 2> dist = vertex - closest_point;
            const double dist2 = dist.dot(dist);
            if (dist2 < line_segment_min_dist2)
            {
                line_segment_min_dist2 = dist2;
                line_segment_min_dist_index = j;
                line_segment_min_dist_t_par = t;
            }
        }
        const CrackSegmentPrecomputed &closest_line_segment = line_segments[line_segment_min_dist_index];
        const Eigen::Vector<double, 2> v0_to_vertex = vertex - (closest_line_segment.v0 +
                                std::clamp(line_segment_min_dist_t_par, 0.0, 1.0) * closest_line_segment.dir);
        const double signed_area = closest_line_segment.dir.x() * v0_to_vertex.y() - v0_to_vertex.x() * closest_line_segment.dir.y();

        int sign = 0;
        int index = -1;
        const double t = line_segment_min_dist_t_par;
        if (t > 1.0f)
        {

            if (line_segment_min_dist_index == line_segments.size() - 1)
            {
                index = line_segments.size() - 1;
                sign = signed_area > 0.0f ? 1 : (signed_area < 0.0f ? -1 : 0);
            }
            else
            {
                index = line_segment_min_dist_index;
                const Eigen::Vector<double, 2>& a = line_segments[line_segment_min_dist_index].dir;
                const Eigen::Vector<double, 2>& b = line_segments[line_segment_min_dist_index + 1].dir;

                const double ab = a.x() * b.y() - a.y() * b.x();
                const double ac = a.x() * v0_to_vertex.y() - a.y() * v0_to_vertex.x();
                const double bc = b.x() * v0_to_vertex.y() - b.y() * v0_to_vertex.x();
                sign = ab < 0 ? (ac > 0 || bc > 0 ? 1 : -1) : (ac > 0 && bc > 0 ? 1 : -1);
            }
        }
        else if (t < 0.0f)
        {
            if (line_segment_min_dist_index == 0)
            {
                index = line_segment_min_dist_index;
                sign = signed_area > 0.0f ? 1 : (signed_area < 0.0f ? -1 : 0);
            }
            else
            {
                index = line_segment_min_dist_index - 1;
                const Eigen::Vector<double, 2>& a = line_segments[line_segment_min_dist_index - 1].dir;
                const Eigen::Vector<double, 2>& b = line_segments[line_segment_min_dist_index].dir;

                const double ab = a.x() * b.y() - a.y() * b.x();
                const double ac = a.x() * v0_to_vertex.y() - a.y() * v0_to_vertex.x();
                const double bc = b.x() * v0_to_vertex.y() - b.y() * v0_to_vertex.x();
                sign = ab < 0 ? (ac > 0 || bc > 0 ? 1 : -1) : (ac > 0 && bc > 0 ? 1 : -1);
            }
        }
        else
        {
            index = line_segment_min_dist_index;
            sign = signed_area > 0.0f ? 1 : (signed_area < 0.0f ? -1 : 0);
        }
        vertices_level_set_signs[i].sign = sign;
        vertices_level_set_signs[i].index = index;
        level_set_1_signed_dist[i] = sign * std::sqrt(line_segment_min_dist2);
        level_set_2_signed_dist[i] =
            (vertex - last_vertex).dot(last_segment.dir) / std::sqrt(last_segment.l_squared);
        level_set_3_signed_dist[i] =
            -(vertex - first_vertex).dot(first_segment.dir) / std::sqrt(last_segment.l_squared);
        vertices_level_set_signs[i].tip = (line_segment_min_dist_index == line_segments.size() - 1 && t > 1.0f)
                                            ? 1
                                            : ((line_segment_min_dist_index == 0 && t < 0.0f) ? -1 : 0);
    }
    return LevelSetFields{
        vertices_level_set_signs,
        level_set_1_signed_dist,
        level_set_2_signed_dist,
        level_set_3_signed_dist
    };
}

inline double crossMagnitudeSigned(const Eigen::Vector<double, 2> &a, const Eigen::Vector<double, 2> &b) noexcept
{
    return a[0] * b[1] - a[1] * b[0];
}

// assume counter-clockwise order of vertices like in standard FEM implementation (ANSYS)
inline bool inQuad(const Eigen::Vector<double, 2>& v00, const Eigen::Vector<double, 2>& v10, const Eigen::Vector<double, 2>& v11, const Eigen::Vector<double, 2>& v01, const Eigen::Vector<double, 2>& p) noexcept
{
    const Eigen::Vector<double, 2> e1 = v10 - v00, e2 = v11 - v10, e3 = v01 - v11, e4 = v00 - v01;
    return crossMagnitudeSigned(p - v00, e1) < 0 && crossMagnitudeSigned(p - v10, e2) < 0 &&
           crossMagnitudeSigned(p - v11, e3) < 0 && crossMagnitudeSigned(p - v01, e4) < 0;
}
// Функция интерполяции точки на ребре
Eigen::Vector<double, 2> interpolate(const Eigen::Vector<double, 2> &p1, const Eigen::Vector<double, 2> &p2, double f1, double f2)
{
    if (f1 == f2)
        return p1; // защита от деления на ноль
    double t = f1 / (f1 - f2);
    return p1 + t * (p2 - p1);
}
inline int positive_mod(int a, int b)
{
    return ((a % b) + b) % b;
}
bool inverseMapping(const Eigen::Vector2d& point, 
                    const std::array<Eigen::Vector2d, 4>& nodes,
                    Eigen::Vector2d& tip_local_coords) {
    double xi = 0, eta = 0;
    ShapeData d;
    for (int iter = 0; iter < 20; ++iter) {
        d.N[0] = 0.25 * (1 - xi) * (1 - eta);
        d.dN_xi_eta(0, 0) = -0.25 * (1 - eta);
        d.dN_xi_eta(1, 0) = -0.25 * (1 - xi);
        // Node 2
        d.N[1] = 0.25 * (1 + xi) * (1 - eta);
        d.dN_xi_eta(0, 1) = 0.25 * (1 - eta);
        d.dN_xi_eta(1, 1) = -0.25 * (1 + xi);
        // Node 3
        d.N[2] = 0.25 * (1 + xi) * (1 + eta);
        d.dN_xi_eta(0, 2) = 0.25 * (1 + eta);
        d.dN_xi_eta(1, 2) = 0.25 * (1 + xi);
        // Node 4
        d.N[3] = 0.25 * (1 - xi) * (1 + eta);
        d.dN_xi_eta(0, 3) = -0.25 * (1 + eta);
        d.dN_xi_eta(1, 3) = 0.25 * (1 - xi);
        
        Eigen::Vector2d iter_coord = Eigen::Vector2d::Zero();
        double dxdxi = 0, dxdet = 0, dydxi = 0, dydet = 0;
        for (int i = 0; i < 4; ++i) {
            iter_coord += d.N[i] * nodes[i];
            dxdxi += d.dN_xi_eta(0, i) * nodes[i].x();
            dydxi += d.dN_xi_eta(0, i) * nodes[i].y();
            dxdet += d.dN_xi_eta(1, i) * nodes[i].x();
            dydet += d.dN_xi_eta(1, i) * nodes[i].y();
        }
        
        Eigen::Vector2d r = point - iter_coord;
        if (r.squaredNorm() < 1e-12 && std::abs(xi) <= 1.0 && std::abs(eta) <= 1.0) {
            tip_local_coords.x() = xi;
            tip_local_coords.y() = eta;
            std::cout << "Newton's method converged after " << iter << " iterations\n";
            return true;
        }
        
        double detJ = dxdxi * dydet - dxdet * dydxi;
        if (fabs(detJ) < 1e-12) return false;
        
        double dxi = ( dydet * r.x() - dxdet * r.y()) / detJ;
        double deta = (-dydxi * r.x() + dxdxi * r.y()) / detJ;
        
        xi += dxi;
        eta += deta;
        
        // optional: early exit if leaving [-2,2] – likely diverging
        if (fabs(xi) > 2.0 || fabs(eta) > 2.0) return false;
    }
    return false; // not converged
}

EnrichedElements find_enriched_elements_by_level_set_fields_simple(const QuadMesh& quad_mesh, const Crack& crack, const LevelSetFields& level_set_fields){
    std::vector<HeavisideEnriched> heaviside_enriched;
    std::vector<TipEnriched> tip_enriched;
    std::vector<int> regular;
    std::vector<bool> heaviside_enriched_nodes;
    std::vector<bool> tip_enriched_nodes;
    heaviside_enriched_nodes.resize(quad_mesh.vertices.size());
    tip_enriched_nodes.resize(quad_mesh.vertices.size());

    if (crack.indices.empty()){
        regular = std::vector<int>(quad_mesh.elements.size());
        std::iota(regular.begin(), regular.end(), 0); 
        return EnrichedElements{regular, heaviside_enriched, tip_enriched, heaviside_enriched_nodes};
    }

    // firstly, find two tip elements
    const Eigen::Vector<double, 2>& crack_tip_1_vertex = crack.vertices[crack.indices[0].v0];
    const Eigen::Vector<double, 2>& crack_tip_2_vertex = crack.vertices[crack.indices[crack.indices.size()-1].v1];
    
    int tip_1_index = -1, tip_2_index = -1;
    if (quad_mesh.elements.size() > std::numeric_limits<int>::max())
        throw std::range_error("quad_mesh.elements.size() > std::numeric_limits<int>::max()");
    for (int i = 0; i < quad_mesh.elements.size(); i++){
        const std::array<int, 4>& element = quad_mesh.elements[i];
        // assume that all elements are convex
        const bool tip1 = inQuad(quad_mesh.vertices[element[0]], quad_mesh.vertices[element[1]], quad_mesh.vertices[element[2]], quad_mesh.vertices[element[3]], crack_tip_1_vertex);
        const bool tip2 = inQuad(quad_mesh.vertices[element[0]], quad_mesh.vertices[element[1]], quad_mesh.vertices[element[2]], quad_mesh.vertices[element[3]], crack_tip_2_vertex);
        if (tip1 && tip2){
            std::cout << "NotImplemented: Both tips in one element." << std::endl;
            continue;
        }
        if (tip1){
            if (tip_1_index != -1) throw std::runtime_error("tip_1_index != -1");
            tip_1_index = i;
            Eigen::Vector<double, 2> intersection_point_local_coords;
            int intersected_edge = -1;
            Eigen::Vector<double, 2> tip_local_coords;
            bool found = false;
            for (int edge = 0; edge < 4; edge++){
                if (level_set_fields.vertices_level_set_signs[element[edge]].tip == -1 && level_set_fields.vertices_level_set_signs[element[edge]].tip == level_set_fields.vertices_level_set_signs[element[(edge + 1)%4]].tip
                && level_set_fields.vertices_level_set_signs[element[edge]].sign*level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign < 0){
                    if (found) throw std::runtime_error("already found");
                    const int opposite_edge = (edge + 2)%4;
                    const double coord = (level_set_fields.level_set_1_signed_dist[element[opposite_edge]]+level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]])/(level_set_fields.level_set_1_signed_dist[element[opposite_edge]]-level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]]); 
                    intersection_point_local_coords = Eigen::Vector<double, 2>{
                        (opposite_edge == 0)*coord+(opposite_edge == 1)*(1)+(opposite_edge == 2)*(-coord)+(opposite_edge == 3)*(-1),
                        (opposite_edge == 1)*coord+(opposite_edge == 2)*(1)+(opposite_edge == 3)*(-coord)+(opposite_edge == 0)*(-1)
                    };
                    inverseMapping(crack_tip_1_vertex, {quad_mesh.vertices[element[0]], quad_mesh.vertices[element[1]], quad_mesh.vertices[element[2]], quad_mesh.vertices[element[3]]}, tip_local_coords);
                    intersected_edge = opposite_edge;
                    found = true;
                    // intersection_point_local_coords = interpolate(quad_mesh.vertices[element[opposite_edge]], quad_mesh.vertices[element[(opposite_edge+1)%4]], level_set_fields.level_set_1_signed_dist[element[opposite_edge]],
                    //                                             level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]]); 
                }
            }
            if (!found) throw std::runtime_error("not found");
            tip_enriched.push_back(TipEnriched{i, intersection_point_local_coords, tip_local_coords,  static_cast<unsigned char>(intersected_edge), 1});
            for (const int node: element){
                tip_enriched_nodes[node] = true;
            }
        }
        if (tip2){
            if (tip_2_index != -1) throw std::runtime_error("tip_2_index != -1");
            tip_2_index = i;
            Eigen::Vector<double, 2> intersection_point_local_coords;
            int intersected_edge = -1;
            Eigen::Vector<double, 2> tip_local_coords;
            bool found = false;
            double min_dist = std::abs(level_set_fields.level_set_1_signed_dist[element[0]]);
            int min_node = 0;
            for (int node = 0; node < 4; node++){
                if (level_set_fields.level_set_1_signed_dist[element[node]] < min_dist){
                    min_node = node;
                    min_dist = std::abs(level_set_fields.level_set_1_signed_dist[element[node]]);
                }
            }
            int min_edge = 0;
            if (level_set_fields.vertices_level_set_signs[element[min_node]].sign > 0){
                min_edge = (min_node + 3 ) % 4;
            }else{
                min_edge = min_node;
            }
            int found_edge = -1;
            for (int edge = 0; edge < 4; edge++){
                if (level_set_fields.vertices_level_set_signs[element[edge]].tip == 1 && level_set_fields.vertices_level_set_signs[element[edge]].tip == level_set_fields.vertices_level_set_signs[element[(edge + 1)%4]].tip
                && level_set_fields.vertices_level_set_signs[element[edge]].sign*level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign < 0
                ){
                // if (level_set_fields.vertices_level_set_signs[element[edge]].sign * level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign < 0){
                // if (level_set_fields.vertices_level_set_signs[element[edge]].sign * level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign < 0
                //     && (level_set_fields.vertices_level_set_signs[element[edge]].tip == -1 || level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].tip == -1)
                // || (edge == min_edge)){
                    found_edge = edge;
                }
            }
            // try another criteria
            if (found_edge == -1){
                found_edge = min_edge;
            }
            if (found_edge != -1){
                if (found) throw std::runtime_error("already found");
                const int opposite_edge = (found_edge + 2)%4;
                const double coord = (level_set_fields.level_set_1_signed_dist[element[opposite_edge]]+level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]])/(level_set_fields.level_set_1_signed_dist[element[opposite_edge]]-level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]]); 
                intersection_point_local_coords = Eigen::Vector<double, 2>{
                    (opposite_edge == 0)*coord+(opposite_edge == 1)*(1)+(opposite_edge == 2)*(-coord)+(opposite_edge == 3)*(-1),
                    (opposite_edge == 1)*coord+(opposite_edge == 2)*(1)+(opposite_edge == 3)*(-coord)+(opposite_edge == 0)*(-1)
                };
                inverseMapping(crack_tip_2_vertex, {quad_mesh.vertices[element[0]], quad_mesh.vertices[element[1]], quad_mesh.vertices[element[2]], quad_mesh.vertices[element[3]]}, tip_local_coords);
                found = true;
                intersected_edge = opposite_edge;
                // intersection_point_local_coords = interpolate(quad_mesh.vertices[element[opposite_edge]], quad_mesh.vertices[element[(opposite_edge+1)%4]], level_set_fields.level_set_1_signed_dist[element[opposite_edge]],
                //                                             level_set_fields.level_set_1_signed_dist[element[(opposite_edge+1)%4]]); 
            }
            if (!found) throw std::runtime_error("not found");
            tip_enriched.push_back(TipEnriched{i, intersection_point_local_coords, tip_local_coords, static_cast<unsigned char>(intersected_edge), 2});
            for (const int node: element){
                tip_enriched_nodes[node] = true;
            }
        }
    }
    // then find heaviside enriched
    bool skip = false;
    for (int i = 0; i < quad_mesh.elements.size(); i++){
        const std::array<int, 4>& element = quad_mesh.elements[i];
        skip = false;
        if ((level_set_fields.vertices_level_set_signs[element[0]].sign == 0 +
            level_set_fields.vertices_level_set_signs[element[1]].sign == 0 +
            level_set_fields.vertices_level_set_signs[element[2]].sign == 0 +
            level_set_fields.vertices_level_set_signs[element[3]].sign == 0 ) > 2)
        {
            std::cerr << "Warning: maybe multiple cracks in one element (0)" << std::endl;
        }
        int crack_within_edge_counter = 0;
        int edge_with_crack = 0;
        for (int edge = 0; edge < 4; edge++){
            if (level_set_fields.vertices_level_set_signs[element[edge]].sign == level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign &&
                level_set_fields.vertices_level_set_signs[element[(edge+1)%4]].sign == 0){
                crack_within_edge_counter++;      
                edge_with_crack = edge;
            }
        }
        if (crack_within_edge_counter != 0){
            if (crack_within_edge_counter > 1){
                std::cerr << "Warning: maybe multiple cracks in one element (01)" << std::endl;
            }else{
                heaviside_enriched_nodes[element[edge_with_crack]] = heaviside_enriched_nodes[element[(edge_with_crack+1)%4]] = 0;
            }
            skip = true;
        }

        // Skip if there is no intersection at the edges; however, do not skip if the sign changes at the opposite nodes.
        else if (level_set_fields.vertices_level_set_signs[element[0]].sign * level_set_fields.vertices_level_set_signs[element[1]].sign >= 0 &&
            level_set_fields.vertices_level_set_signs[element[1]].sign * level_set_fields.vertices_level_set_signs[element[2]].sign >= 0 &&
            level_set_fields.vertices_level_set_signs[element[2]].sign * level_set_fields.vertices_level_set_signs[element[3]].sign >= 0 &&
            level_set_fields.vertices_level_set_signs[element[3]].sign * level_set_fields.vertices_level_set_signs[element[0]].sign >= 0 &&
            level_set_fields.vertices_level_set_signs[element[0]].sign * level_set_fields.vertices_level_set_signs[element[2]].sign >= 0 &&
            level_set_fields.vertices_level_set_signs[element[1]].sign * level_set_fields.vertices_level_set_signs[element[3]].sign >= 0
        )
        {
            skip = true;
        }
        else if(
                level_set_fields.vertices_level_set_signs[element[0]].sign * level_set_fields.vertices_level_set_signs[element[2]].sign >= 0 &&
                (level_set_fields.vertices_level_set_signs[element[1]].tip != 0 || level_set_fields.vertices_level_set_signs[element[3]].tip != 0)  ||
                level_set_fields.vertices_level_set_signs[element[1]].sign * level_set_fields.vertices_level_set_signs[element[3]].sign >= 0 &&
                (level_set_fields.vertices_level_set_signs[element[0]].tip != 0 || level_set_fields.vertices_level_set_signs[element[2]].tip != 0) != 0 
            )
        { 
            skip = true;
        }
        else if ((level_set_fields.vertices_level_set_signs[element[0]].tip == -1 && level_set_fields.vertices_level_set_signs[element[1]].tip == -1 &&
                        level_set_fields.vertices_level_set_signs[element[3]].tip == -1 &&
                        level_set_fields.vertices_level_set_signs[element[2]].tip == -1) ||
                        (level_set_fields.vertices_level_set_signs[element[0]].tip == 1 && level_set_fields.vertices_level_set_signs[element[1]].tip == 1 &&
                        level_set_fields.vertices_level_set_signs[element[3]].tip == 1 && level_set_fields.vertices_level_set_signs[element[2]].tip == 1))
        {
            skip = true;
        }
        else if ((level_set_fields.vertices_level_set_signs[element[0]].tip == -1) + (level_set_fields.vertices_level_set_signs[element[1]].tip == -1) +
                            (level_set_fields.vertices_level_set_signs[element[3]].tip == -1) +
                            (level_set_fields.vertices_level_set_signs[element[2]].tip == -1) >=
                        2)
        {
            skip = true;
        }
        else if ((level_set_fields.vertices_level_set_signs[element[0]].tip == 1) + (level_set_fields.vertices_level_set_signs[element[1]].tip == 1) +
                            (level_set_fields.vertices_level_set_signs[element[3]].tip == 1) +
                            (level_set_fields.vertices_level_set_signs[element[2]].tip == 1) >=
                        2)
        {
            skip = true;
        }
        else if (i == tip_1_index || i == tip_2_index)
        {
            skip = true;
        }
        if (skip)
        {
            if (!(i == tip_1_index || i == tip_2_index)){
                regular.push_back(i);
            }
            continue;
        }
        int intersection_count = 0;
        std::array<Eigen::Vector<double, 2>, 2> intersection_points_local_coords;
        std::array<int, 2> intersected_edges;
        for (int edge = 0; edge < 4; edge++){
            if (level_set_fields.level_set_1_signed_dist[element[edge]] == 0 || level_set_fields.level_set_1_signed_dist[element[edge]] * level_set_fields.level_set_1_signed_dist[element[(edge+1)%4]] < 0)
            {
                if (intersection_count >= 2)
                {
                    throw std::runtime_error("NotImplemented: more than 2 intersection_edges");
                }
                const double coord = (level_set_fields.level_set_1_signed_dist[element[edge]]+level_set_fields.level_set_1_signed_dist[element[(edge+1)%4]])/(level_set_fields.level_set_1_signed_dist[element[edge]]-level_set_fields.level_set_1_signed_dist[element[(edge+1)%4]]); 
                intersection_points_local_coords[intersection_count] = Eigen::Vector<double, 2>{
                    coord * (edge==0) + (1) * (edge==1) + (-coord) * (edge==2) + (-1) * (edge==3),
                    coord * (edge==1) + (1) * (edge==2) + (-coord) * (edge==3) + (-1) * (edge==0)
                };
                intersected_edges[intersection_count] = edge;
                intersection_count++;
            }
        }
        if (intersection_count == 0) 
        {
            continue;
            throw std::runtime_error("NotImplemented: 0 intersections");
        }
        if (intersection_count == 1) 
        {
            intersected_edges[1] = intersected_edges[0];
            intersection_points_local_coords[1] = intersection_points_local_coords[0];
            std::cerr << "Warning: 1 intersection" << std::endl;
        }
        if (intersection_count > 2) 
        {
            throw std::runtime_error("NotImplemented: more than 2 intersection_edges");
        }
        heaviside_enriched.push_back(HeavisideEnriched{i, intersection_points_local_coords, intersected_edges});
        for (const int& node: element){
            heaviside_enriched_nodes[node] = 1;
        }
    }

    return EnrichedElements{regular, heaviside_enriched, tip_enriched, heaviside_enriched_nodes, tip_enriched_nodes};
}



EnrichedElementsTriangulation triangulate_enriched(const QuadMesh& quad_mesh, const EnrichedElements& enriched_elements, const LevelSetFields& level_set_fields){
    std::vector<HeavisideTriangulation> hvsd_trng;
    std::array<unsigned char, 5> poly;
    unsigned char polyVerticesCount = 0;
    for (int i = 0; i < enriched_elements.heaviside_enriched.size(); i++){
        const HeavisideEnriched& heaviside_enriched = enriched_elements.heaviside_enriched[i];
        const std::array<int, 4> element = quad_mesh.elements[heaviside_enriched.id];
        
        // Heaviside positive triangles first
        std::array<std::array<unsigned char, 3>, 4> tri_indices;
        short triangles_num = 0;
        short positive_heaviside_triangles_num = 0;
        polyVerticesCount = 0;
        
        // the entire polygon is above the crack
        if (heaviside_enriched.intersected_edges[0] == heaviside_enriched.intersected_edges[1]){
            polyVerticesCount = 4;
            poly = {0, 1, 2, 3};
            triangles_num = 2;
            int sign = 0;
            int current_sign;
            // iterate through the each vertex of the polygon and check for sign equality
            int zero_sign_vertex_count = 0;
            bool non_zero_found = false;
            bool all_signs_are_equal = true;
            for (unsigned char j = 0; j < polyVerticesCount; j++)
            {
                current_sign = level_set_fields.vertices_level_set_signs[element[poly[j]]].sign;
                if (current_sign != 0) {
                    if (non_zero_found && sign != current_sign){
                        all_signs_are_equal = false;
                    }
                    sign = current_sign;
                    non_zero_found = true;
                }else{
                    zero_sign_vertex_count++;
                }
            }
            if (sign == 0) throw std::runtime_error("sign is still zero");
            // TODO: add algorithm to find crack intersection with edges 
            if (!all_signs_are_equal){
                std::cerr << "Warning: not are signs are equal. Maybe multiple cracks in one element" << std::endl;
            }
            if (zero_sign_vertex_count > 2){
                std::cerr << "Warning: zero_sign_vertex_count > 2. Maybe multiple cracks in one element" << std::endl;
            }
            tri_indices[0] = {0, 1, 2};
            tri_indices[1] = {0, 2, 3};
            positive_heaviside_triangles_num = sign > 0 ? 2 : 0;
            hvsd_trng.push_back(HeavisideTriangulation{triangles_num, positive_heaviside_triangles_num, tri_indices});
            continue;
        }
        short triangle_num = 4;
        for (unsigned char edge = 0; edge < 4; edge++){
            if (heaviside_enriched.intersected_edges[0] == edge){
                if (level_set_fields.vertices_level_set_signs[element[edge]].sign != 0){
                    poly[polyVerticesCount++] = edge;
                    poly[polyVerticesCount++] = 4;
                }else{
                    poly[polyVerticesCount++] = edge;
                    triangle_num--;
                }
                edge = heaviside_enriched.intersected_edges[1];
                if (level_set_fields.vertices_level_set_signs[element[edge]].sign != 0){
                    poly[polyVerticesCount++] = 5;
                }else{
                    poly[polyVerticesCount++] = edge;
                    triangle_num--;
                }
            }else{
                poly[polyVerticesCount++] = edge;
            }
        }
        if (polyVerticesCount == 3)
        {
            int sign = 0;
            bool all_signs_are_equal = true;
            int zero_sign_vertex_count = 0;
            for (int j = 0; j < 3; j++){
                if (poly[j] < 4){
                    if (level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != 0){
                        if (sign != 0 && level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != sign)
                            all_signs_are_equal = false;
                        sign = level_set_fields.vertices_level_set_signs[element[poly[j]]].sign;
                    }else{
                        zero_sign_vertex_count++;
                    }
                }
            }
            if (sign == 0) throw std::runtime_error("sign is still zero  (1)");
            if (!all_signs_are_equal){
                std::cerr << "Warning: not are signs are equal. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (zero_sign_vertex_count > 2){
                std::cerr << "Warning: zero_sign_vertex_count > 2. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (sign > 0){
                positive_heaviside_triangles_num = 1;
                tri_indices[0] = {poly[0], poly[1], poly[2]};
            }else{
                // Turned out that first pass is single negative triangle
                positive_heaviside_triangles_num = triangle_num - 1;
                tri_indices[positive_heaviside_triangles_num] = {poly[0], poly[1], poly[2]};
            }
        }
        else
        {
            int sign = 0;
            bool all_signs_are_equal = true;
            int zero_sign_vertex_count = 0;
            for (int j = 0; j < polyVerticesCount; j++){
                if (poly[j] < 4){
                    if (level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != 0){
                        if (sign != 0 && level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != sign)
                            all_signs_are_equal = false;
                        sign = level_set_fields.vertices_level_set_signs[element[poly[j]]].sign;
                    }else{
                        zero_sign_vertex_count++;
                    }
                }
            }
            if (sign == 0) throw std::runtime_error("sign is still zero  (1)");
            if (!all_signs_are_equal){
                std::cerr << "Warning: not are signs are equal. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (zero_sign_vertex_count > 2){
                std::cerr << "Warning: zero_sign_vertex_count > 2. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (sign > 0){
                // There are more than one positive triangles
                positive_heaviside_triangles_num = polyVerticesCount - 2;
                for (unsigned char j = 1; j < polyVerticesCount - 1; j++)
                {
                    tri_indices[j-1] = {poly[0], poly[j], poly[static_cast<unsigned char>(j + 1)]};
                }
            }else{
                // triangle_num - (polyVerticesCount - 2)
                positive_heaviside_triangles_num = triangle_num - (polyVerticesCount - 2);
                for (unsigned char j = 1; j < polyVerticesCount - 1; j++)
                {
                    tri_indices[j+positive_heaviside_triangles_num-1] = {poly[0], poly[j], poly[static_cast<unsigned char>(j + 1)]};
                }
            }
        }
        // now we go below crack
        polyVerticesCount = 0;
        if (level_set_fields.vertices_level_set_signs[element[heaviside_enriched.intersected_edges[1]]].sign != 0){
            poly[polyVerticesCount++] = 5;
        }else{
            poly[polyVerticesCount++] = heaviside_enriched.intersected_edges[1];
        }
        if (level_set_fields.vertices_level_set_signs[element[heaviside_enriched.intersected_edges[0]]].sign != 0){
            poly[polyVerticesCount++] = 4;
        }else{
            poly[polyVerticesCount++] = heaviside_enriched.intersected_edges[0];
        }
        for (int edge = heaviside_enriched.intersected_edges[0] + 1; edge < 4; edge++)
        {
            if (heaviside_enriched.intersected_edges[1] == edge)
            {
                if (level_set_fields.vertices_level_set_signs[element[heaviside_enriched.intersected_edges[0]]].sign != 0){
                    poly[polyVerticesCount++] = edge;
                }
                break;
            }
            else
            {
                poly[polyVerticesCount++] = edge;
            }
        }
        if (polyVerticesCount == 3)
        {
            int sign = 0;
            bool all_signs_are_equal = true;
            int zero_sign_vertex_count = 0;
            for (int j = 0; j < polyVerticesCount; j++){
                if (poly[j] < 4){
                    if (level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != 0){
                        if (sign != 0 && level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != sign)
                            all_signs_are_equal = false;
                        sign = level_set_fields.vertices_level_set_signs[element[poly[j]]].sign;
                    }else{
                        zero_sign_vertex_count++;
                    }
                }
            }
            if (sign == 0) throw std::runtime_error("sign is still zero  (1)");
            if (!all_signs_are_equal){
                std::cerr << "Warning: not are signs are equal. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (zero_sign_vertex_count > 2){
                std::cerr << "Warning: zero_sign_vertex_count > 2. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (sign > 0){
                tri_indices[0] = {poly[0], poly[1], poly[2]};
            }else{
                tri_indices[positive_heaviside_triangles_num] = {poly[0], poly[1], poly[2]};
            }
        }
        else
        {
            int sign = 0;
            bool all_signs_are_equal = true;
            int zero_sign_vertex_count = 0;
            for (int j = 0; j < polyVerticesCount; j++){
                if (poly[j] < 4){
                    if (level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != 0){
                        if (sign != 0 && level_set_fields.vertices_level_set_signs[element[poly[j]]].sign != sign)
                            all_signs_are_equal = false;
                        sign = level_set_fields.vertices_level_set_signs[element[poly[j]]].sign;
                    }else{
                        zero_sign_vertex_count++;
                    }
                }
            }
            if (sign == 0) throw std::runtime_error("sign is still zero  (1)");
            if (!all_signs_are_equal){
                std::cerr << "Warning: not are signs are equal. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (zero_sign_vertex_count > 2){
                std::cerr << "Warning: zero_sign_vertex_count > 2. Maybe multiple cracks in one element (1)" << std::endl;
            }
            if (sign > 0){
                for (unsigned char j = 0; j < positive_heaviside_triangles_num; j++)
                {
                    tri_indices[j] = {poly[0], poly[j+1], poly[static_cast<unsigned char>(j + 2)]};
                }
            }else{
                for (unsigned char j = 0; j < triangle_num - positive_heaviside_triangles_num; j++)
                {
                    tri_indices[j+positive_heaviside_triangles_num] = {poly[0], poly[j+1], poly[static_cast<unsigned char>(j + 2)]};
                }
            }
        }
        hvsd_trng.push_back(HeavisideTriangulation{triangle_num, positive_heaviside_triangles_num, tri_indices});
    }
    std::vector<TipTriangulation> tip_trng;
    for (int i = 0; i < enriched_elements.tip_enriched.size(); i++){
        const TipEnriched& tip_enriched = enriched_elements.tip_enriched[i];
        const std::array<int, 4> element = quad_mesh.elements[tip_enriched.id];
        std::array<std::array<unsigned char, 3>, 5> tri_indices;
        int triangles_count = 0;
        const unsigned char intersected_edge = static_cast<unsigned char>(tip_enriched.intersected_edge);
        // 0...3 is vertices of element, 4 is for point on edge, 5 is for tip
        tri_indices[0] = {5, intersected_edge, 4};
        tri_indices[1] = {5, 4, static_cast<unsigned char>(positive_mod(intersected_edge + 1, 4))};
        for (unsigned int j = 0; j < 3; j++)
        {
            tri_indices[2+j] = {5,  static_cast<unsigned char>(positive_mod(intersected_edge + j + 1, 4)),
            static_cast<unsigned char>(positive_mod(intersected_edge + j + 2, 4))};
        }
        tip_trng.push_back(TipTriangulation{tri_indices});
    }
    return EnrichedElementsTriangulation{hvsd_trng, tip_trng};
}

