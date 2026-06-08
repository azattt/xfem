#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <variant>
#include <vector>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <Eigen/Dense>
#include <Eigen/src/SVD/JacobiSVD.h>

#include "fem.h"
#include "gui.h"
#include "levelset.h"
#include "postprocess.h"

#include "BC.h"

#include "misc.h"

#include "camera.h"
#include "shader.h"

class RainbowColormap
{
  public:
    // ANSYS classic rainbow colors (blue -> cyan -> green -> yellow -> red)
    static glm::vec3 getColor(float t)
    {
        // Clamp t to [0, 1] range
        t = glm::clamp(t, 0.0f, 1.0f);

        // ANSYS rainbow typically has 5 segments:
        // 0.0: Blue (0,0,1)
        // 0.25: Cyan (0,1,1)
        // 0.5: Green (0,1,0)
        // 0.75: Yellow (1,1,0)
        // 1.0: Red (1,0,0)

        if (t < 0.25f)
        {
            // Blue to Cyan
            float localT = t / 0.25f;
            return glm::vec3(0.0f,
                             localT, // Green increases
                             1.0f    // Blue constant
            );
        }
        else if (t < 0.5f)
        {
            // Cyan to Green
            float localT = (t - 0.25f) / 0.25f;
            return glm::vec3(0.0f,
                             1.0f,         // Green constant
                             1.0f - localT // Blue decreases
            );
        }
        else if (t < 0.75f)
        {
            // Green to Yellow
            float localT = (t - 0.5f) / 0.25f;
            return glm::vec3(localT, // Red increases
                             1.0f,   // Green constant
                             0.0f);
        }
        else
        {
            // Yellow to Red
            float localT = (t - 0.75f) / 0.25f;
            return glm::vec3(1.0f,          // Red constant
                             1.0f - localT, // Green decreases
                             0.0f);
        }
    }

    // Get color with alpha for transparency
    static glm::vec4 getColorRGBA(float t, float alpha = 1.0f)
    {
        glm::vec3 rgb = getColor(t);
        return glm::vec4(rgb, alpha);
    }
};

struct Triangle
{
    GLuint v0, v1, v2;
};

struct Circle
{
    glm::vec3 center_and_radius;
    glm::vec4 color;
};
struct Rectangle
{
    glm::vec4 bbox;
    glm::vec4 color;
};
struct Quad
{
    glm::vec2 v00, v10, v11, v01;
    uint32_t color;
};

struct Vertex
{
    glm::vec2 position;
    uint32_t colorPacked; // 4 байта: RGBA
};
uint32_t packColor(const glm::vec4 &color)
{
    auto to_uint8 = [](float c) -> uint32_t {
        // Clamp to [0,1] and round to nearest integer
        float clamped = std::clamp(c, 0.0f, 1.0f);
        return static_cast<uint32_t>(clamped * 255.0f + 0.5f) & 0xFF;
    };
    uint32_t r = to_uint8(color.r);
    uint32_t g = to_uint8(color.g);
    uint32_t b = to_uint8(color.b);
    uint32_t a = to_uint8(color.a);

    return (r << 0) | (g << 8) | (b << 16) | (a << 24);
}
// struct PolygonalChainClosed
// {
//     std::vector<glm::vec2> points;
//     glm::vec4 color;
// };

// settings
unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.5f, 0.5f, 2.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

bool lmbPressed = false;
bool rmbPressed = false;

// timing
float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

bool draw = true;

void windowRefreshCallback([[maybe_unused]] GLFWwindow *window)
{
    draw = true;
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(UP, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(DOWN, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(LEFT, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(RIGHT, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(FORWARD, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(BACKWARD, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback([[maybe_unused]] GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    if (width != 0 && height != 0)
    {
        SCR_WIDTH = width;
        SCR_HEIGHT = height;
        glViewport(0, 0, width, height);
        draw = true;
    }
    else
    {
        draw = false;
    }
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback([[maybe_unused]] GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // float xoffset = xpos - lastX;
    // float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    // camera.ProcessMouseMovement(xoffset, yoffset);
}

void mouse_button_callback([[maybe_unused]] GLFWwindow *window, int button, int action, [[maybe_unused]] int mods)
{
    if (button == GLFW_MOUSE_BUTTON_1)
    {
        if (action == GLFW_PRESS)
        {
            lmbPressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            lmbPressed = false;
        }
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback([[maybe_unused]] GLFWwindow *window, [[maybe_unused]] double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

int main()
{
    std::cout << "CPU has AVX2: " << hasAVX2() << std::endl;
    std::ifstream mesh_data("mesh/mesh.txt");
    if (!mesh_data.is_open()){
        throw std::runtime_error("Couldn't open mesh/mesh.txt");
    } 
    double w, h;
    int wn, hn;
    double scale = 0; // factor to make deformation visible

    mesh_data >> w >> h >> wn >> hn >> scale;
    // scale = 0;
    // double w = 1, h = 1;
    // int wn = 2, hn = 3;
    double wh = w / (wn - 1), hh = h / (hn - 1);
    std::vector<LinearQuad::Element> elements;

    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    // seed = 1773229899;
    std::cout << "seed: " << seed << std::endl;
    std::srand(seed);

    // for (int i = 0; i < 5; i++){
    //     crack_vertices.push_back(glm::vec2{0.2f-0.1f*glm::cos(2*glm::pi<float>()/5*i),
    //     0.1f*glm::sin(2*glm::pi<float>()/5*i) + 0.5f}); crack_indices.push_back(Line{static_cast<unsigned
    //     int>(crack_vertices.size() - 2), static_cast<unsigned int>(crack_vertices.size() - 1)});
    // }
    // crack_vertices.push_back(glm::vec2{0.1f, 0.5f});
    // for (int i = 2; i < 10; i++)
    // {
    //     // crack_vertices.push_back(glm::vec2{(float)(rand() % 100) / 100, (float)(rand() % 100) / 100});
    //     // crack_vertices.push_back(glm::vec2{(float)i / 10, glm::sin((float)2 * i / 5) / 4 + 0.5f});
    //     // crack_vertices.push_back(glm::vec2{(float)i / 10, (float)(rand() % 100) / 1000 + 0.5f});
    //     crack_vertices.push_back(glm::vec2{(float)i / 10, 0.5f});
    //     // crack_vertices.push_back(glm::vec2{glm::cos((float)2 * i / 5) / 4 + 0.5f, glm::sin((float)2 * i / 5) / 4 +
    //     // 0.5f});
    //     crack_indices.push_back(Line{static_cast<unsigned int>(crack_vertices.size() - 2),
    //                                  static_cast<unsigned int>(crack_vertices.size() - 1)});
    // }

    // crack_vertices.push_back(glm::vec2{0.0f, 0.5f});
    // crack_vertices.push_back(glm::vec2{0.5f, 0.5f});
    // crack_indices.push_back(Line{0, 1});

    // through node
    // crack_vertices.push_back(glm::vec2{0.6f, 0.2f});
    // crack_vertices.push_back(glm::vec2{0.9f, 0.3f});

    int total_vertices = wn * hn;
    QuadMesh mesh;
    mesh.vertices = std::vector<Eigen::Vector<double, 2>>();
    mesh.vertices.reserve(total_vertices);

    for (int j = 0; j < hn; j++)
    {
        for (int i = 0; i < wn; i++)
        {
            mesh.vertices.push_back(Eigen::Vector<double, 2>{i * wh, j * hh});
            // mesh.vertices.push_back(Eigen::Vector<double, 2>{i * wh + (rand() % 100) / 1000.0, j * hh + (rand() % 100) / 1000.0});
        }
    }

    mesh.elements = std::vector<std::array<int, 4>>();
    mesh.elements.reserve((hn - 1) * (wn - 1));
    for (int j = 0; j < hn - 1; j++)
    {
        for (int i = 0; i < wn - 1; i++)
        {
            // if ((i >= wn/2 -1 && i <= wn/2) && j == hn/2) continue;
            mesh.elements.push_back(
                std::array<int, 4>{j * wn + i, j * wn + i + 1, (j + 1) * wn + i + 1, (j + 1) * wn + i});
        }
    }
    
    Crack crack;

    std::ifstream crack_data("mesh/crack.txt");
    if (!crack_data.is_open()){
        throw std::runtime_error("Couldn't open mesh/crack.txt");
    } 
    double a, b;
    while (crack_data >> a >> b) {
        crack.vertices.push_back(Eigen::Vector2d(a, b));
    }
    if (crack.vertices.size() < 2) {
        throw std::runtime_error("Crack must contain at least two points");
    }
    for (int i = 1; i < crack.vertices.size(); i++){
        crack.indices.push_back(CrackSegment{i-1, i});
    }
    
    std::ifstream interaction_integral_data("mesh/interaction_integral.txt");
    if (!interaction_integral_data.is_open()){
        throw std::runtime_error("Couldn't open mesh/interaction_integral.txt");
    }
    double Rin, Rout;
    interaction_integral_data >> Rin >> Rout;

    bool disable_output = true;
    std::ifstream disable_output_data("mesh/disable_output.txt");
    if (!disable_output_data.is_open()){
        std::cerr << "Couldn't open mesh/disable_output.txt. disable_output set to true." << std::endl;
    }else{
        disable_output_data >> disable_output;
    }
    bool disable_debug_output = true;
    std::ifstream disable_debug_output_data("mesh/disable_debug_output.txt");
    if (!disable_debug_output_data.is_open()){
        std::cerr << "Couldn't open mesh/disable_debug_output.txt. disable_debug_output set to true." << std::endl;
    }else{
        disable_debug_output_data >> disable_debug_output;
    }
    Eigen::Vector2d first_crack_segment = (crack.vertices[crack.indices[0].v0] -
    crack.vertices[crack.indices[0].v1]);
    Eigen::Vector2d crack_tip_1_t = first_crack_segment.normalized();
    Eigen::Vector2d crack_tip_1_n = Eigen::Vector2d{-crack_tip_1_t.y(), crack_tip_1_t.x()};

    Eigen::Vector2d last_crack_segment = (crack.vertices[crack.indices[crack.indices.size()-1].v1] -
    crack.vertices[crack.indices[crack.indices.size()-1].v0]);
    Eigen::Vector2d crack_tip_2_t = last_crack_segment.normalized();
    Eigen::Vector2d crack_tip_2_n = Eigen::Vector2d{-crack_tip_2_t.y(), crack_tip_2_t.x()};

    
    find_enriched_elements(mesh, crack);
    LevelSetFields level_set_fields = compute_level_set_fields(mesh, crack);
    EnrichedElements enriched_elements =
        find_enriched_elements_by_level_set_fields_simple(mesh, crack, level_set_fields);
    // EnrichedElements enriched_elements;
    // EnrichedElementsTriangulation enriched_elements_triangulation;
    EnrichedElementsTriangulation enriched_elements_triangulation =
        triangulate_enriched(mesh, enriched_elements, level_set_fields);

    std::vector<unsigned int> node_offset(total_vertices);
    std::vector<unsigned int> node_ndof(total_vertices);
    unsigned int dof_counter = 0;
    for (unsigned int n = 0; n < total_vertices; ++n)
    {
        node_offset[n] = dof_counter;
        node_ndof[n] = 12; // always two standard DOFs
        // if (enriched_elements.heaviside_enriched_nodes[n])
        // {
        //     node_ndof[n] += 2;
        // }
        // if (enriched_elements.tip_enriched_nodes[n])      node_ndof[n] += 8;
        dof_counter += node_ndof[n];
    }
    std::vector<bool> active(dof_counter);
    // After node_offset and node_ndof are ready:
    size_t total_triplets = 0;
    for (const int element_id : enriched_elements.regular)
    {
        int n_local = 0;
        for (int i = 0; i < 4; ++i)
        { // assuming 4 nodes per element
            int node = mesh.elements[element_id][i];
            n_local += node_ndof[node];
        }
        total_triplets += n_local * (n_local + 1) / 2;
    }
    std::vector<Eigen::Triplet<double>> triplets;
    std::cout << "Not enriched " << enriched_elements.regular.size()
              << " Heaviside: " << enriched_elements.heaviside_enriched.size()
              << " Tip enriched: " << enriched_elements.tip_enriched.size() << std::endl;
    std::cout << "Assembling matrix of size: " << dof_counter << std::endl;
    std::cout << "Triplets count: " << total_triplets << std::endl;

    const double E = 2 * std::pow(10, 11);
    constexpr double nu = 0.3;
    constexpr double t = 0.1;
    const Eigen::Matrix3d D = setup_D_matrix(E, nu, true);

    // tip elements
    // TODO: Not forget to rotate, because crack is rotated atan is incorrect!!!.
    for (int i = 0; i < enriched_elements.tip_enriched.size(); i++)
    {
        const auto &enriched_element = enriched_elements.tip_enriched[i];
        const std::array<int, 4> &element = mesh.elements[enriched_element.id];
        const TipTriangulation &triangulation = enriched_elements_triangulation.tip_enriched_triangulation[i];
        // u_x u_y f1_x f1_y f2_x f2_y f3_x f3_y f4_x f4_y . total 10 dof per node
        // 4 nodes. total 40 dofs per element
        Eigen::Matrix<double, 40, 40> Ke;

        Ke.setZero();
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
            //     std::cout << "Triangle " << i << " vertices (local): "
            //   << local_points[triangle[0]].x << "," << local_points[triangle[0]].y << " ; "
            //   << local_points[triangle[1]].x << "," << local_points[triangle[1]].y << " ; "
            //   << local_points[triangle[2]].x << "," << local_points[triangle[2]].y << "\n";
            for (unsigned int gp = 0; gp < LinearTriangle::Triangle13PointRule::NGauss; gp++)
            {
                B.setZero();
                double r = LinearTriangle::Triangle13PointRule::gauss_pts[gp][0];
                double s = LinearTriangle::Triangle13PointRule::gauss_pts[gp][1];
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
                factor =
                    LinearTriangle::Triangle13PointRule::gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
                if (det_tri < 0)
                    std::cout << "det_tri < 0" << std::endl;
                if (jd.detJ < 0)
                    std::cout << "jd.detJ < 0" << std::endl;
                Ke += factor * (B.transpose() * D * B) * t;
                total_area += LinearTriangle::Triangle13PointRule::gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
                if (!disable_debug_output){
                Eigen::VectorXd rigid_x(40);
                rigid_x.setZero();
                for (int i = 0; i < 4; ++i) rigid_x(10*i) = 1.0;   // standard u_x = 1
                    Eigen::VectorXd strain = B * rigid_x;   // but B is not available after assembly; we need to compute
                    // it again or store
                    // Instead, compute the residual Ke * rigid_x
                    Eigen::VectorXd res = Ke * rigid_x;
                
                    std::cout << "Norm of Ke * rigid_x: " << res.norm() << std::endl;  // should be near 0
                }
            }
        }
        // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 16, 16>> es(Ke);
        // std::cout << "eigenvalues: " << es.eigenvalues() << std::endl;
        // std::cout << "total_area: " << total_area << std::endl;
        // std::cout << "det: " << Ke.determinant() << std::endl;
        
        if (enriched_elements.heaviside_enriched_nodes[element[0]] || enriched_elements.heaviside_enriched_nodes[element[1]] || enriched_elements.heaviside_enriched_nodes[element[2]] ||
            enriched_elements.heaviside_enriched_nodes[element[3]] ){
            Eigen::Matrix<double, 48, 48> Ke_expanded;
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    for (int k = 0; k < 10; k++)
                    {
                        for (int l = 0; l < 10; l++)
                        {
                            const int nodes_per_elem = 4;
                            const int dofs_per_node_local = 10;      // standard+tip
                            const int dofs_per_node_global = 12;     // standard+heaviside+tip
                            const int offset_tip = 2;                // skip Heaviside DOFs
                            // Map local DOF index to global block index
                            int global_row_dof;
                            int global_col_dof;

                            if (k < 2)  // standard DOFs
                                global_row_dof = k;                // 0,1
                            else        // tip DOFs
                                global_row_dof = k + offset_tip;   // 2->4, 3->5, ..., 9->11

                            if (l < 2)
                                global_col_dof = l;
                            else
                                global_col_dof = l + offset_tip;
                            Ke_expanded(12 * i + global_row_dof, 12 * j + global_col_dof)
                            = Ke(10 * i + k, 10 * j + l);
                            
                            // Ke_expanded(12 * i + 0, 12 * j + 0) = Ke(10 * i + 0, 10 * j + 0);
                            // Ke_expanded(12 * i + 0, 12 * j + 1) = Ke(10 * i + 0, 10 * j + 1);
                            // Ke_expanded(12 * i + 1, 12 * j + 0) = Ke(10 * i + 1, 10 * j + 0);
                            // Ke_expanded(12 * i + 1, 12 * j + 1) = Ke(10 * i + 1, 10 * j + 1);
                        }
                    }
                }
            }
            // std::cout << "output 1: " << Ke_expanded;
            FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{element[0], element[1], element[2], element[3]},
                                                    Ke_expanded, triplets, node_offset, node_ndof, 12, active);
            // std::cout << "expanded\n" << Ke_expanded << std::endl;
            // std::cout << "not expanded\n" << Ke << std::endl;
            // std::ofstream tmp_file("tip_enriched.txt");
            // tmp_file << Ke_expanded;
            // tmp_file.close();
            // std::ofstream tmp_file2("tip_enriched2.txt");
            // tmp_file2 << Ke;
            // tmp_file2.close();
        }else{
            Eigen::Matrix<double, 48, 48> Ke_expanded;
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    for (int k = 0; k < 10; k++)
                    {
                        for (int l = 0; l < 10; l++)
                        {
                            const int nodes_per_elem = 4;
                            const int dofs_per_node_local = 10;      // standard+tip
                            const int dofs_per_node_global = 12;     // standard+heaviside+tip
                            const int offset_tip = 2;                // skip Heaviside DOFs
                            // Map local DOF index to global block index
                            int global_row_dof;
                            int global_col_dof;

                            if (k < 2)  // standard DOFs
                                global_row_dof = k;                // 0,1
                            else        // tip DOFs
                                global_row_dof = k + offset_tip;   // 2->4, 3->5, ..., 9->11

                            if (l < 2)
                                global_col_dof = l;
                            else
                                global_col_dof = l + offset_tip;
                            Ke_expanded(12 * i + global_row_dof, 12 * j + global_col_dof)
                            = Ke(10 * i + k, 10 * j + l);
                            
                            // Ke_expanded(12 * i + 0, 12 * j + 0) = Ke(10 * i + 0, 10 * j + 0);
                            // Ke_expanded(12 * i + 0, 12 * j + 1) = Ke(10 * i + 0, 10 * j + 1);
                            // Ke_expanded(12 * i + 1, 12 * j + 0) = Ke(10 * i + 1, 10 * j + 0);
                            // Ke_expanded(12 * i + 1, 12 * j + 1) = Ke(10 * i + 1, 10 * j + 1);
                        }
                    }
                }
            }
            FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{element[0], element[1], element[2], element[3]},
                                                    Ke_expanded, triplets, node_offset, node_ndof, 12, active);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 48, 48>> es(Ke_expanded);
            if (!disable_debug_output){
                std::cout << "eigenvalues of tip enriched: " << es.eigenvalues() << std::endl;
                std::cout << "total_area: " << total_area << std::endl;
                std::cout << "det: " << Ke.determinant() << std::endl;
            }
            // std::cout << Ke << std::endl;
        }
    }

    // heaviside elements
    for (int i = 0; i < enriched_elements.heaviside_enriched.size(); i++)
    {
        const auto &enriched_element = enriched_elements.heaviside_enriched[i];
        const std::array<int, 4> &element = mesh.elements[enriched_element.id];
        const HeavisideTriangulation &triangulation =
            enriched_elements_triangulation.heaviside_enriched_triangulation[i];
        Eigen::Matrix<double, 16, 16> Ke;

        Ke.setZero();
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
                                                       enriched_element.intersection_points_local_coords[0],
                                                       enriched_element.intersection_points_local_coords[1]};
        int sign = 1;
        const Eigen::Matrix<double, 4, 2> coords{{{points[0].x(), points[0].y()},
                                                  {points[1].x(), points[1].y()},
                                                  {points[2].x(), points[2].y()},
                                                  {points[3].x(), points[3].y()}}};
        std::array<int, 4> node_signs;
        for (int n = 0; n < 4; ++n)
        {
            node_signs[n] = level_set_fields.vertices_level_set_signs[element[n]].sign;
        }
        // Eigen::Matrix2f J_xy_xieta = shape.dN_xi_eta * coords;
        double total_area = 0.0;
        for (unsigned int j = 0; j < triangulation.triangles_num; j++)
        {

            const std::array<unsigned char, 3> &triangle = triangulation.tri_indices[j];
            if (j >= triangulation.positive_heaviside_triangles_num)
            {
                sign = -1;
            }
            Eigen::Matrix2d J_xieta_rs{{{local_points[triangle[1]].x() - local_points[triangle[0]].x(),
                                         local_points[triangle[2]].x() - local_points[triangle[0]].x()},
                                        {local_points[triangle[1]].y() - local_points[triangle[0]].y(),
                                         local_points[triangle[2]].y() - local_points[triangle[0]].y()}}};
            double det_tri = J_xieta_rs.determinant();
            //     std::cout << "Triangle " << i << " vertices (local): "
            //   << local_points[triangle[0]].x << "," << local_points[triangle[0]].y << " ; "
            //   << local_points[triangle[1]].x << "," << local_points[triangle[1]].y << " ; "
            //   << local_points[triangle[2]].x << "," << local_points[triangle[2]].y << "\n";
            for (unsigned int gp = 0; gp < LinearTriangle::Triangle13PointRule::NGauss; gp++)
            {
                double r = LinearTriangle::Triangle13PointRule::gauss_pts[gp][0];
                double s = LinearTriangle::Triangle13PointRule::gauss_pts[gp][1];
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
                // Initialize to zero
                // std::cout << coords << std::endl;
                // std::cout << shape.dN_xi_eta << std::endl;
                jd.J = shape.dN_xi_eta * coords;
                bool invertible;
                jd.J.computeInverseAndDetWithCheck(jd.invJ, jd.detJ, invertible, 1e-12);
                if (!invertible)
                    throw std::runtime_error("Jacobi matrix is not invertible");
                Eigen::Matrix<double, 2, 4> dN_dx_dy;
                dN_dx_dy = jd.invJ * shape.dN_xi_eta;
                Eigen::Matrix<double, 3, 16> B;
                for (int i = 0; i < 4; ++i)
                {
                    B(0, 4 * i) = dN_dx_dy(0, i); // du/dx
                    B(0, 4 * i + 1) = 0;
                    B(1, 4 * i) = 0;
                    B(1, 4 * i + 1) = dN_dx_dy(1, i); // dv/dy
                    B(2, 4 * i) = dN_dx_dy(1, i);     // dv/dx
                    B(2, 4 * i + 1) = dN_dx_dy(0, i); // du/dy

                    double shifted_factor = sign - node_signs[i];      // = 0 on same side, ±2 on opposite
                    B(0, 4 * i + 2) = dN_dx_dy(0, i) * shifted_factor; // du/dx
                    B(0, 4 * i + 3) = 0;
                    B(1, 4 * i + 2) = 0;
                    B(1, 4 * i + 3) = dN_dx_dy(1, i) * shifted_factor; // dv/dy
                    B(2, 4 * i + 2) = dN_dx_dy(1, i) * shifted_factor; // dv/dx
                    B(2, 4 * i + 3) = dN_dx_dy(0, i) * shifted_factor; // du/dy
                }
                double factor =
                    LinearTriangle::Triangle13PointRule::gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
                if (det_tri < 0)
                    std::cout << "det_tri < 0" << std::endl;
                if (jd.detJ < 0)
                    std::cout << "jd.detJ < 0" << std::endl;
                Ke += factor * (B.transpose() * D * B) * t;
                total_area += det_tri;
                // Eigen::VectorXd rigid_x(16);
                // rigid_x.setZero();
                // for (int i = 0; i < 4; ++i) rigid_x(4*i) = 1.0;   // standard u_x = 1
                // Eigen::VectorXd strain = B * rigid_x;   // but B is not available after assembly; we need to compute
                // it again or store
                // // Instead, compute the residual Ke * rigid_x
                // Eigen::VectorXd res = Ke * rigid_x;
                // std::cout << "Norm of Ke * rigid_x: " << res.norm() << std::endl;  // should be near 0
            }
        }
        if (!disable_debug_output){
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 16, 16>> es(Ke);
            std::cout << "eigenvalues of heaviside element: " << es.eigenvalues() << std::endl;
            std::cout << "total_area: " << total_area << std::endl;
            std::cout << "det: " << Ke.determinant() << std::endl;
            std::cout << element[0] << " " << element[1] << " " << element[2] << " " << element[3] << std::endl;
        }
        if (true || enriched_elements.tip_enriched_nodes[element[0]] || enriched_elements.tip_enriched_nodes[element[1]] || enriched_elements.tip_enriched_nodes[element[2]] ||
            enriched_elements.tip_enriched_nodes[element[3]]){
            Eigen::Matrix<double, 48, 48> Ke_expanded;
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        for (int l = 0; l < 4; l++)
                        {
                            Ke_expanded(12 * i + k, 12 * j + l) =
                             Ke(4 * i + k, 4 * j + l);
                            // Ke_expanded(12 * i + 0, 12 * j + 0) = Ke(10 * i + 0, 10 * j + 0);
                            // Ke_expanded(12 * i + 0, 12 * j + 1) = Ke(10 * i + 0, 10 * j + 1);
                            // Ke_expanded(12 * i + 1, 12 * j + 0) = Ke(10 * i + 1, 10 * j + 0);
                            // Ke_expanded(12 * i + 1, 12 * j + 1) = Ke(10 * i + 1, 10 * j + 1);
                        }
                    }
                }
                // std::ofstream tmp_file("test_heaviside.txt");
                // tmp_file << Ke_expanded;
                // tmp_file.close();
                // std::ofstream tmp_file2("test_heaviside2.txt");
                // tmp_file2 << Ke;
                // tmp_file2.close();
            }
            FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{element[0], element[1], element[2], element[3]},
                                                        Ke_expanded, triplets, node_offset, node_ndof, 12, active);
        }else{
            
            FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{element[0], element[1], element[2], element[3]},
                                                        Ke, triplets, node_offset, node_ndof, 4, active);
        }
    }

    Eigen::Matrix<double, 8, 8> Ke;
    Eigen::Matrix<double, 16, 16> Ke_expanded;
    Eigen::Matrix<double, 4, 2> coordMat;
    unsigned int elementsTotal = mesh.elements.size();
    unsigned int elementsCreated = 0;

    // regular elements
    unsigned int percent = 0, lastPercent = 0;
    for (const int element_id : enriched_elements.regular)
    {
        const std::array<int, 4> element = mesh.elements[element_id];
        percent = static_cast<unsigned int>(100.0 * elementsCreated / elementsTotal);
        if (percent > lastPercent)
        { // чтобы не выводить одно и то же значение много раз
            std::cout << percent << "% ";
            lastPercent = percent;
        }
        for (int i = 0; i < 4; ++i)
        {
            coordMat(i, 0) = mesh.vertices[element[i]].x();
            coordMat(i, 1) = mesh.vertices[element[i]].y();
        }
        Ke = LinearQuad::element_stiffness(coordMat, D, t);
        // whether no enrichment or blend
        if (false && (enriched_elements.heaviside_enriched_nodes[element[0]] ||
            enriched_elements.heaviside_enriched_nodes[element[1]] ||
            enriched_elements.heaviside_enriched_nodes[element[2]] ||
            enriched_elements.heaviside_enriched_nodes[element[3]]))
        {
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    Ke_expanded(4 * i + 0, 4 * j + 0) = Ke(2 * i + 0, 2 * j + 0);
                    Ke_expanded(4 * i + 0, 4 * j + 1) = Ke(2 * i + 0, 2 * j + 1);
                    Ke_expanded(4 * i + 1, 4 * j + 0) = Ke(2 * i + 1, 2 * j + 0);
                    Ke_expanded(4 * i + 1, 4 * j + 1) = Ke(2 * i + 1, 2 * j + 1);
                }
            }
            FEMAssemble::addElementSparseUpperStiffness(element, Ke_expanded, triplets, node_offset, node_ndof, 4, active);
        }
        else if (true || enriched_elements.tip_enriched_nodes[element[0]] || enriched_elements.tip_enriched_nodes[element[1]] || enriched_elements.tip_enriched_nodes[element[2]] ||
            enriched_elements.tip_enriched_nodes[element[3]]){
            Eigen::Matrix<double, 48, 48> Ke_expanded;
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        for (int l = 0; l < 2; l++)
                        {
                            Ke_expanded(12 * i + k, 12 * j + l) =
                             Ke(2 * i + k, 2 * j + l);
                            // Ke_expanded(12 * i + 0, 12 * j + 0) = Ke(10 * i + 0, 10 * j + 0);
                            // Ke_expanded(12 * i + 0, 12 * j + 1) = Ke(10 * i + 0, 10 * j + 1);
                            // Ke_expanded(12 * i + 1, 12 * j + 0) = Ke(10 * i + 1, 10 * j + 0);
                            // Ke_expanded(12 * i + 1, 12 * j + 1) = Ke(10 * i + 1, 10 * j + 1);
                        }
                    }
                }
            }
            FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{element[0], element[1], element[2], element[3]},
                                                        Ke_expanded, triplets, node_offset, node_ndof, 12, active);
        }
        else
        {
            if (!disable_debug_output)
                std::cout << element[0] << " " << element[1] << " " << element[2] << " " << element[3] << std::endl;
            FEMAssemble::addElementSparseUpperStiffness(element, Ke, triplets, node_offset, node_ndof, 2, active);
        }
        elementsCreated++;
    }
    if (!disable_debug_output)
        std::cout << std::endl;
    auto K = FEMAssemble::createStiffnessFromTriplets(triplets, dof_counter);


    std::cout << "Global stiffness matrix is assembled" << std::endl;
    if (!disable_output)
    {
        std::cout << "Writing to file." << std::endl;
        Eigen::MatrixXd spK = Eigen::MatrixXd(K);
        std::ofstream KoutFile("K.txt");
        KoutFile << spK;
    }

    std::cout << "Creating RHS" << std::endl;
    Eigen::VectorXd P(dof_counter);
    P.setZero();

    std::cout << "Applying boundary conditions" << std::endl;
    std::vector<int> fixedDofs;
    std::vector<double> fixedValues;

    // for (unsigned int i = 0; i < wn; i++)
    // {
    //     int off = node_offset[i];
    //     fixedDofs.push_back(off);
    //     fixedDofs.push_back(off + 1);
    //     fixedValues.push_back(0);
    //     fixedValues.push_back(0);
    //     off = node_offset[(wn * (hn - 1) + i)];
    //     // fixedDofs.push_back(off);
    //     // fixedDofs.push_back(off + 1);
    //     // fixedValues.push_back(0.1);
    //     // fixedValues.push_back(0.2);
    //     // FEMAssemble::fixDOF(K, P, i, UX|UY);
    // }
    // for (int i = 0; i < wn; i++)
    // {
    //     int off = node_offset[(wn * (hn - 1) + i)];
    //     P(off + 1) = 10000000;
    // }
    // P(10) = 1; 
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd{Eigen::MatrixXd(K)};;
    // double cond = svd.singularValues()(0) 
    // / svd.singularValues()(svd.singularValues().size()-1);
    // std::cout << "Condition number of KBC: " << cond << std::endl;
    // for (unsigned int i = hn * 0.2; i < hn * 0.8; i++){
    //     P((wn*(i)+0)*2) = -10000;
    //     P((wn*(i)+wn-1)*2) = 10000;
    // }
    applyBC(w, h, wn, hn, P, node_offset, fixedDofs, fixedValues);

    FEMAssemble::applyDirichletSymmetric(K, P, fixedDofs, fixedValues);
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
    for (int i = 0; i < dof_counter; ++i) {
        if (!active[i]) {
            // DOF never used – fix to zero
            K.coeffRef(i,i) = 1.0;
            P(i) = 0.0;
        }
    }
    std::cout << "Solving linear system" << std::endl;
    Eigen::VectorXd u = FEMAssemble::solveSparseSPDUpper(K, P);
    // Eigen::MatrixXd Kdense = K;
    // Eigen::MatrixXd diff = Kdense - Kdense.transpose();
    // // std::cout << diff << std::endl;
    // std::cout << "Max asymmetry: " << diff.maxCoeff() << std::endl;
    std::cout << "Energy: " << 0.5 * u.dot(K.selfadjointView<Eigen::Upper>() * u) << std::endl;
    // std::cout << "Ku - P: " << K*u - P << std::endl;
    // Eigen::VectorXd residual = K * u - P;
    Eigen::VectorXd residual = K.selfadjointView<Eigen::Upper>() * u - P;
    double residual_norm = residual.norm();
    // std::cout << "Ku - P: " << residual << std::endl;
    std::cout << "||Ku - P|| = " << residual_norm << std::endl;
    std::cout << "Writing result to file" << std::endl;
    std::ostringstream Uout;
    Uout << u;
    std::ofstream UoutFile("U.txt");
    UoutFile << Uout.str();
    // Eigen::VectorXd u;
    // u.setZero(dof_counter);
    std::cout << "Post-processing" << std::endl;
    

    std::cout << "Scaling results with factor: " << scale << std::endl;
    std::vector<Eigen::Vector2d> node_displacement(total_vertices);
    for (int n = 0; n < total_vertices; ++n)
    {
        int off = node_offset[n];
        // Standard DOFs are the first two at offset 'off' and 'off+1'
        node_displacement[n].x() = u(off);
        node_displacement[n].y() = u(off + 1);
    }
    std::vector<Eigen::Vector2d> vertices_displaced = mesh.vertices;
    for (int idx = 0; idx < total_vertices; ++idx)
    {
        vertices_displaced[idx] +=
            Eigen::Vector2d(node_displacement[idx].x() * scale, node_displacement[idx].y() * scale);
    }

    glm::vec2 v1 = toGlm(vertices_displaced[0].cast<float>().eval()),
              v3 = toGlm(vertices_displaced[vertices_displaced.size() - 1].cast<float>().eval());
    camera.Position = glm::vec3{(v1 + v3) / 2.0f, 1.0f};
    camera.Zoom = 90.0f;
    camera.MovementSpeed = 0.5f;

    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6); // use 4.6
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    checkGLSLVersion();
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "Current working directory: " << cwd << std::endl;

    std::vector<Circle> circles;
    std::vector<Rectangle> rectangles;
    std::vector<Quad> quads;
    std::vector<PolygonalChain> polygonal_chains;

    

    // for (const int element_id : enriched_elements.regular)
    // {
    //         const LinearQuad::Element &element = mesh.elements[element_id];
    //         quads.push_back(Quad{toGlm(vertices_displaced[element[0]].cast<float>().eval()),
    //                             toGlm(vertices_displaced[element[1]].cast<float>().eval()),
    //                             toGlm(vertices_displaced[element[2]].cast<float>().eval()),
    //                             toGlm(vertices_displaced[element[3]].cast<float>().eval()),
    //                             packColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f))});
    // }

    for (const auto& element: mesh.elements)
    {
            quads.push_back(Quad{toGlm(vertices_displaced[element[0]].cast<float>().eval()),
                                toGlm(vertices_displaced[element[1]].cast<float>().eval()),
                                toGlm(vertices_displaced[element[2]].cast<float>().eval()),
                                toGlm(vertices_displaced[element[3]].cast<float>().eval()),
                                packColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f))});
    }

    drawHeavisideElements(enriched_elements.heaviside_enriched, mesh, enriched_elements_triangulation.heaviside_enriched_triangulation,
    level_set_fields.vertices_level_set_signs, u, node_offset, enriched_elements.heaviside_enriched_nodes, scale);
    drawTipElements(enriched_elements.tip_enriched, mesh, u, node_offset, enriched_elements.heaviside_enriched_nodes, enriched_elements.tip_enriched_nodes, scale, polygonal_chains,
crack_tip_1_t, crack_tip_1_n, crack_tip_2_t, crack_tip_2_n);
computeStress<13>(
    enriched_elements.tip_enriched,
    enriched_elements.heaviside_enriched,
    mesh,
    enriched_elements_triangulation.tip_enriched_triangulation,
    enriched_elements_triangulation.heaviside_enriched_triangulation,
    u,
    node_offset,
    enriched_elements.heaviside_enriched_nodes,
    enriched_elements.tip_enriched_nodes,
    level_set_fields.vertices_level_set_signs,
    LinearTriangle::Triangle13PointRule::gauss_pts,
    LinearTriangle::Triangle13PointRule::gauss_wts,
    crack_tip_1_t,
    crack_tip_1_n,
    crack_tip_2_t,
    crack_tip_2_n,
    D,
    2 * std::pow(10, 11),
    0.3,
    Rin,
    Rout
);

    unsigned int idx;
    for (int i = 0; i < wn; i++)
    {
        for (int j = 0; j < hn; j++)
        {
            idx = j * wn + i;
            circles.push_back(Circle{glm::vec3{toGlm(vertices_displaced[idx].cast<float>().eval()), 1.0f / SCR_WIDTH},
                                     glm::vec4{0.0f, 1.0f, 0.0f, 1.0f}});
        }
    }
    // stress
    std::vector<Eigen::Vector3d> nodalStress(mesh.vertices.size(), Eigen::Vector3d::Zero());
    std::vector<int> nodalCount(mesh.vertices.size(), 0);   
    for (const int element_id : enriched_elements.regular)
    {
        const std::array<int, 4> element = mesh.elements[element_id];

    }
    PolygonalChain crack_chain;
    
    crack_chain.color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
    
    for (const Eigen::Vector2d vertex: crack.vertices){
        crack_chain.points.push_back(glm::vec2(vertex.x(), vertex.y()));
    }
    polygonal_chains.push_back(crack_chain);

    Shader xfem_shader("shaders/xfem.vert", "shaders/xfem.frag");

    std::vector<Vertex> chain_vertices;
    std::vector<GLint> firstsStrip;
    std::vector<GLsizei> countsStrip;
    std::vector<GLint> firstsLoop;
    std::vector<GLsizei> countsLoop;

    Shader chain_program("shaders/pchain.vert", "shaders/pchain.frag");
    for (const PolygonalChain &chain : polygonal_chains)
    {
        if (chain.points.empty())
            continue;

        size_t start = chain_vertices.size();
        uint32_t packedColor = packColor(chain.color);

        for (const auto &pt : chain.points)
        {
            chain_vertices.push_back({pt, packedColor});
        }

        firstsStrip.push_back(static_cast<GLint>(start));
        countsStrip.push_back(static_cast<GLsizei>(chain.points.size()));
    }

    GLuint chainVAO, chainVBO;
    glGenVertexArrays(1, &chainVAO);
    glGenBuffers(1, &chainVBO);

    glBindVertexArray(chainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, chainVBO);
    glBufferData(GL_ARRAY_BUFFER, chain_vertices.size() * sizeof(Vertex), chain_vertices.data(), GL_STATIC_DRAW);

    // Атрибут 0: позиция (2 float)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);

    // Атрибут 1: цвет (упакованный GL_UNSIGNED_BYTE, нормализованный)
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void *)offsetof(Vertex, colorPacked));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    Shader circle_shader("shaders/circle.vert", "shaders/circle.frag");
    GLuint circleVerticesVBO, circleVBO, circleVAO;
    constexpr int circles_vertices_number = 22;
    if (circles.size())
    {
        constexpr float PI = glm::pi<float>();
        std::vector<glm::vec2> circle_vertices;
        circle_vertices.resize(circles_vertices_number);
        circle_vertices[0].x = 0.0f;
        circle_vertices[0].y = 0.0f;
        for (int i = 1; i < circles_vertices_number; i++)
        {
            circle_vertices[i].x = glm::cos(2 * PI / (circles_vertices_number - 2) * (i - 1));
            circle_vertices[i].y = glm::sin(2 * PI / (circles_vertices_number - 2) * (i - 1));
        }

        glGenBuffers(1, &circleVerticesVBO);
        glGenBuffers(1, &circleVBO);
        glGenVertexArrays(1, &circleVAO);

        glBindVertexArray(circleVAO);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, circleVerticesVBO);
        glBufferData(GL_ARRAY_BUFFER, (circles_vertices_number) * 2 * sizeof(float), &circle_vertices[0],
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
        glBufferData(GL_ARRAY_BUFFER, circles.size() * sizeof(Circle), &circles[0], GL_STATIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Circle), (void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Circle), (void *)sizeof(glm::vec3));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(2, 1);
    }
    Shader rectangle_shader("shaders/rect.vert", "shaders/rect.frag");
    GLuint rectangleVerticesVBO, rectangleVerticesEBO, rectangleVBO, rectangleVAO;
    if (rectangles.size())
    {

        std::vector<glm::vec2> rectangle_vertices;
        rectangle_vertices.push_back(glm::vec2(0.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 1.0f));
        rectangle_vertices.push_back(glm::vec2(0.0f, 1.0f));

        std::vector<GLuint> rectangle_indices = {
            0, 1, 3, // first triangle
            1, 2, 3  // second triangle
        };

        glGenBuffers(1, &rectangleVerticesVBO);
        glGenBuffers(1, &rectangleVerticesEBO);
        glGenBuffers(1, &rectangleVBO);
        glGenVertexArrays(1, &rectangleVAO);

        glBindVertexArray(rectangleVAO);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, rectangleVerticesVBO);
        glBufferData(GL_ARRAY_BUFFER, rectangle_vertices.size() * sizeof(glm::vec2), rectangle_vertices.data(),
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rectangleVerticesEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, rectangle_indices.size() * sizeof(GLuint), rectangle_indices.data(),
                     GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, rectangleVBO);
        glBufferData(GL_ARRAY_BUFFER, rectangles.size() * sizeof(Rectangle), &rectangles[0], GL_STATIC_DRAW);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Rectangle), (void *)0);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, rectangleVBO);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Rectangle), (void *)sizeof(glm::vec4));
        glVertexAttribDivisor(2, 1);
    }
    Shader quad_shader("shaders/quad.vert", "shaders/quad.frag");
    GLuint quadVerticesVBO, quadVBO, quadVAO;

    if (quads.size())
    {
        std::vector<glm::vec2> rectangle_vertices;
        rectangle_vertices.push_back(glm::vec2(0.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 1.0f));
        rectangle_vertices.push_back(glm::vec2(0.0f, 1.0f));

        glGenBuffers(1, &quadVerticesVBO);
        glGenBuffers(1, &quadVBO);
        glGenVertexArrays(1, &quadVAO);
        glBindVertexArray(quadVAO);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quadVerticesVBO);
        glBufferData(GL_ARRAY_BUFFER, rectangle_vertices.size() * sizeof(glm::vec2), &rectangle_vertices[0],
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, quads.size() * sizeof(Quad), &quads[0], GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)0);
        glVertexAttribDivisor(1, 1);
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v10));
        glVertexAttribDivisor(2, 1);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v11));
        glVertexAttribDivisor(3, 1);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v01));
        glVertexAttribDivisor(4, 1);
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Quad), (void *)offsetof(Quad, color));
        glVertexAttribDivisor(5, 1);
    }

    TriangleGUI::Renderer::instance().initializeGL();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----

        processInput(window);
        if (!draw)
        {
            glfwWaitEvents();
            continue;
        }

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);
        glm::mat4 projection =
            glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        xfem_shader.use();
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 MVP = projection * view;
        // xfem_shader.setMat4("mvp", MVP);
        // xfem_shader.setVec4("color", 1.0f, 0.0f, 0.0f, 1.0f);
        // // glBindVertexArray(VAO);
        // // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // // glDrawElements(GL_TRIANGLES, indices.size() * 3, GL_UNSIGNED_INT, (void *)0);
        // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        // if (rectangles.size())
        // {
        //     rectangle_shader.use();
        //     rectangle_shader.setMat4("mvp", MVP);
        //     glBindVertexArray(rectangleVAO);
        //     glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, rectangles.size());
        // }

        quad_shader.use();
        quad_shader.setMat4("mvp", MVP);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, quads.size());

        if (circles.size())
        {
            circle_shader.use();
            circle_shader.setMat4("mvp", MVP);
            glBindVertexArray(circleVAO);
            glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, circles_vertices_number, circles.size());
        }
        // xfem_shader.use();
        // xfem_shader.setMat4("mvp", MVP);
        // xfem_shader.setVec4("color", 0.0f, 1.0f, 1.0f, 1.0f);
        // glBindVertexArray(lineVAO);
        // glDrawElements(GL_LINES, crack_indices.size() * 2, GL_UNSIGNED_INT, (void *)0);

        chain_program.use();
        chain_program.setMat4("mvp", MVP);
        glBindVertexArray(chainVAO);

        if (!firstsStrip.empty())
        {
            glMultiDrawArrays(GL_LINE_STRIP, firstsStrip.data(), countsStrip.data(), firstsStrip.size());
        }
        if (!firstsLoop.empty())
        {
            glMultiDrawArrays(GL_LINE_LOOP, firstsLoop.data(), countsLoop.data(), firstsLoop.size());
        }

        TriangleGUI::Renderer::instance().draw(MVP);
        glBindVertexArray(0);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */

        // std::cout << 1.0 / deltaTime << std::endl;
        draw = false;
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
