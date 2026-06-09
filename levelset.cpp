#include "levelset.h"

#include <cmath>
#include <limits>
#include <numeric> // std::iota

#include "gui.h"

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

static double cross2d(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b
)
{
    return a.x() * b.y() - a.y() * b.x();
}

static bool segmentIntersection2D(
    const Eigen::Vector2d& P,
    const Eigen::Vector2d& Q,
    const Eigen::Vector2d& A,
    const Eigen::Vector2d& B,
    double& s,
    double& t
)
{
    const Eigen::Vector2d r = Q - P;
    const Eigen::Vector2d e = B - A;

    const double denom = cross2d(r, e);

    if (std::abs(denom) < 1e-14)
    {
        return false;
    }

    const Eigen::Vector2d AP = A - P;

    s = cross2d(AP, e) / denom;
    t = cross2d(AP, r) / denom;

    const double eps = 1e-10;

    return s > eps &&
           s <= 1.0 + eps &&
           t >= -eps &&
           t <= 1.0 + eps;
}

struct TipExitEdgeResult
{
    int edge = -1;
    Eigen::Vector2d point_global = Eigen::Vector2d::Zero();
};
static TipExitEdgeResult findTipExitEdgeByGeometry(
    const QuadMesh& mesh,
    const std::array<int, 4>& element,
    const Eigen::Vector2d& tip_point,
    const Eigen::Vector2d& next_crack_point,
    int tip_index
)
{
    TipExitEdgeResult result;

    double best_s = 1e100;

    for (int edge = 0; edge < 4; ++edge)
    {
        const int n0 = edge;
        const int n1 = (edge + 1) % 4;

        const Eigen::Vector2d A =
            mesh.vertices[element[n0]];

        const Eigen::Vector2d B =
            mesh.vertices[element[n1]];

        double s = 0.0;
        double t = 0.0;

        const bool hit =
            segmentIntersection2D(
                tip_point,
                next_crack_point,
                A,
                B,
                s,
                t
            );

        if (!hit)
        {
            continue;
        }

        if (s < best_s)
        {
            best_s = s;
            result.edge = edge;
            result.point_global =
                tip_point + s * (next_crack_point - tip_point);
        }
    }

    if (result.edge < 0)
    {
        std::cout << "Cannot find exit edge geometrically.\n";
        std::cout << "tip_index = " << tip_index << "\n";
        std::cout << "tip_point = " << tip_point.transpose() << "\n";
        std::cout << "next_crack_point = " << next_crack_point.transpose() << "\n";
        std::cout << "segment length = "
                  << (next_crack_point - tip_point).norm()
                  << "\n";

        throw std::runtime_error(
            "Cannot find unique exit edge for tip element: "
            "crack segment may be too short and may not cross element boundary"
        );
    }

    return result;
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
            if (tip_1_index != -1) {
                throw std::runtime_error("tip_1_index != -1");
            }

            tip_1_index = i;

            Eigen::Vector2d intersection_point_local_coords;
            Eigen::Vector2d tip_local_coords;
            unsigned char intersected_edge = 0;

            const CrackSegment& first_segment = crack.indices.front();

            const Eigen::Vector2d tip_point =
                crack.vertices[first_segment.v0];

            const Eigen::Vector2d next_crack_point =
                crack.vertices[first_segment.v1];

            TipExitEdgeResult exit =
                findTipExitEdgeByGeometry(
                    quad_mesh,
                    element,
                    tip_point,
                    next_crack_point,
                    1
                );
        
            const bool ok_inverse = inverseMapping(
                tip_point,
                {quad_mesh.vertices[element[0]],
                quad_mesh.vertices[element[1]],
                quad_mesh.vertices[element[2]],
                quad_mesh.vertices[element[3]]},
                tip_local_coords
            );

            if (!ok_inverse) {
                throw std::runtime_error("inverseMapping failed for tip 1");
            }

            tip_enriched.push_back(
                TipEnriched{
                    i,
                    intersection_point_local_coords,
                    tip_local_coords,
                    intersected_edge,
                    1
                }
            );

            for (const int node : element) {
                tip_enriched_nodes[node] = true;
            }
        }
        if (tip2){
            if (tip_2_index != -1) {
                throw std::runtime_error("tip_2_index != -1");
            }

            tip_2_index = i;

            Eigen::Vector2d intersection_point_local_coords;
            Eigen::Vector2d tip_local_coords;
            unsigned char intersected_edge = 0;

           const CrackSegment& last_segment = crack.indices.back();

            const Eigen::Vector2d tip_point =
                crack.vertices[last_segment.v1];

            const Eigen::Vector2d next_crack_point =
                crack.vertices[last_segment.v0];

            TipExitEdgeResult exit =
                findTipExitEdgeByGeometry(
                    quad_mesh,
                    element,
                    tip_point,
                    next_crack_point,
                    2
                );

            const bool ok_inverse = inverseMapping(
                tip_point,
                {quad_mesh.vertices[element[0]],
                quad_mesh.vertices[element[1]],
                quad_mesh.vertices[element[2]],
                quad_mesh.vertices[element[3]]},
                tip_local_coords
            );

            if (!ok_inverse) {
                throw std::runtime_error("inverseMapping failed for tip 2");
            }

            tip_enriched.push_back(
                TipEnriched{
                    i,
                    intersection_point_local_coords,
                    tip_local_coords,
                    intersected_edge,
                    2
                }
            );

            for (const int node : element) {
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

