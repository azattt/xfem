#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <variant>
#include <vector>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <Eigen/Dense>

#include "fem.h"

#include "camera.h"
#include "shader.h"

inline float crossMagnitudeSigned(const glm::vec2 &a, const glm::vec2 &b) noexcept
{
    return a.x * b.y - a.y * b.x;
}

bool inQuad(glm::vec2 v00, glm::vec2 v10, glm::vec2 v11, glm::vec2 v01, glm::vec2 p)
{
    glm::vec2 e1 = v10 - v00, e2 = v11 - v10, e3 = v01 - v11, e4 = v00 - v01;
    return crossMagnitudeSigned(p - v00, e1) < 0 && crossMagnitudeSigned(p - v10, e2) < 0 &&
           crossMagnitudeSigned(p - v11, e3) < 0 && crossMagnitudeSigned(p - v01, e4) < 0;
}

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

// struct QuadTree{
//     QuadTree NW, NE, SE, SW;

// };

// Функция интерполяции точки на ребре
glm::vec2 interpolate(const glm::dvec2 &p1, const glm::dvec2 &p2, double f1, double f2)
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

struct Triangle
{
    GLuint v0, v1, v2;
};
struct Line
{
    GLuint v0, v1;
};
struct LineSegment
{
    glm::vec2 v0;
    glm::vec2 dir;
    float l_squared;
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
struct PolygonalChain
{
    std::vector<glm::vec2> points;
    glm::vec4 color;
};
// struct PolygonalChainClosed
// {
//     std::vector<glm::vec2> points;
//     glm::vec4 color;
// };
struct LevelSetSign
{
    int sign;
    int tip;
    unsigned int index;
};

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
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
    glViewport(0, 0, width, height);
    draw = true;
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

inline bool segmentIntersect(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d, float &t, float &u)
{
    glm::vec2 ab = b - a;
    glm::vec2 cd = d - c;
    glm::vec2 ac = c - a;

    float denom = ab.x * cd.y - ab.y * cd.x; // cross(ab, cd)
    if (glm::abs(denom) < 1e-10f)
        return false; // parallel

    float inv = 1.0f / denom;
    t = (ac.x * cd.y - ac.y * cd.x) * inv; // cross(ac, cd)
    u = (ac.x * ab.y - ac.y * ab.x) * inv; // cross(ac, ab)

    return t >= 0.f && t <= 1.f && u >= 0.f && u <= 1.f;
}

int main()
{
    std::vector<Circle> circles;
    std::vector<Rectangle> rectangles;
    std::vector<Quad> quads;
    float w = 1, h = 1;
    unsigned int wn = 40, hn = 40;
    float wh = w / (wn - 1), hh = h / (hn - 1);

    std::vector<glm::vec2> vertices;
    // std::vector<Triangle> indices;

    unsigned int total_vertices = wn * hn;
    vertices.resize(total_vertices);

    for (unsigned int j = 0; j < hn; j++)
    {
        for (unsigned int i = 0; i < wn; i++)
        {
            vertices[j * wn + i] = glm::vec2{i * wh, j * hh};
        }
    }

    std::vector<LinearQuad::Element> elements;
    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

    // GLuint VBO, EBO, VAO;
    // glGenBuffers(1, &VBO);
    // glGenBuffers(1, &EBO);
    // glGenVertexArrays(1, &VAO);

    // glBindVertexArray(VAO);

    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec2), vertices.data(), GL_STATIC_DRAW);

    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(Triangle), indices.data(), GL_STATIC_DRAW);

    // glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    // glEnableVertexAttribArray(0);

    // glBindVertexArray(0);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "Current working directory: " << cwd << std::endl;
    Shader xfem_shader("shaders/xfem.vert", "shaders/xfem.frag");

    xfem_shader.use();

    GLuint lineVBO, lineEBO, lineVAO;
    glGenBuffers(1, &lineVBO);
    glGenBuffers(1, &lineEBO);
    glGenVertexArrays(1, &lineVAO);

    glBindVertexArray(lineVAO);
    glLineWidth(3.0f);
    
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    seed = 1773229899;
    std::cout << "seed: " << seed << std::endl;
    std::srand(seed);
    std::vector<glm::vec2> crack_vertices;
    std::vector<Line> crack_indices;
    std::vector<LineSegment> line_segments;
    if (crack_vertices.size() > std::numeric_limits<GLuint>::max())
    {
        throw std::out_of_range("crack_vertices.size() size is bigger than unsigned int");
    }
    
    // for (int i = 0; i < 5; i++){
        //     crack_vertices.push_back(glm::vec2{0.2f-0.1f*glm::cos(2*glm::pi<float>()/5*i),
        //     0.1f*glm::sin(2*glm::pi<float>()/5*i) + 0.5f}); crack_indices.push_back(Line{static_cast<unsigned
        //     int>(crack_vertices.size() - 2), static_cast<unsigned int>(crack_vertices.size() - 1)});
        // }
    crack_vertices.push_back(glm::vec2{0.1f, 0.5f});
    for (int i = 2; i < 10; i++)
    {
        // crack_vertices.push_back(glm::vec2{(float)(rand() % 100) / 100, (float)(rand() % 100) / 100});
        // crack_vertices.push_back(glm::vec2{(float)i / 10, glm::sin((float)2 * i / 5) / 4 + 0.5f});
        // crack_vertices.push_back(glm::vec2{(float)i / 10, (float)(rand() % 100) / 1000 + 0.5f});
        crack_vertices.push_back(glm::vec2{(float)i / 10, 0.5f});
        // crack_vertices.push_back(glm::vec2{glm::cos((float)2 * i / 5) / 4 + 0.5f, glm::sin((float)2 * i / 5) / 4 +
        // 0.5f});
        crack_indices.push_back(Line{static_cast<unsigned int>(crack_vertices.size() - 2),
                                     static_cast<unsigned int>(crack_vertices.size() - 1)});
    }

    line_segments.reserve(crack_indices.size());
    glm::vec2 segment;
    for (const Line &line : crack_indices)
    {
        segment = crack_vertices[line.v1] - crack_vertices[line.v0];
        line_segments.push_back(LineSegment{crack_vertices[line.v0], segment, glm::dot(segment, segment)});
    }
    // crack_vertices.push_back(glm::vec2{0.0f, 0.5f});
    // crack_vertices.push_back(glm::vec2{0.5f, 0.5f});
    // crack_indices.push_back(Line{0, 1});

    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, crack_vertices.size() * sizeof(glm::vec2), crack_vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineEBO);

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, crack_indices.size() * sizeof(Line), crack_indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    glEnableVertexAttribArray(0);

    std::vector<PolygonalChain> polygonal_chains;

    glm::vec2 v0_to_vertex;
    glm::vec2 dist;
    float t, signed_area;
    size_t line_segment_min_dist_index = 0;
    float line_segment_min_dist2;
    float line_segment_min_dist_t_par;
    glm::vec2 closest_point;


    std::vector<bool> heaviside_enriched_nodes(vertices.size(), false);

    std::vector<LevelSetSign> vertices_level_set_signs;
    std::vector<double> level_set_1_signed_dist;
    std::vector<double> level_set_2_signed_dist;
    std::vector<double> level_set_3_signed_dist;
    std::vector<int> level_set_tips;

    vertices_level_set_signs.resize(total_vertices);
    level_set_1_signed_dist.resize(total_vertices);
    level_set_2_signed_dist.resize(total_vertices);
    level_set_3_signed_dist.resize(total_vertices);
    level_set_tips.resize(total_vertices);

    float ab, ac, bc;
    int sign, tip;
    unsigned int index;
    float dist2;

    LineSegment last_segment = line_segments[line_segments.size() - 1];
    glm::vec2 last_vertex = last_segment.v0 + last_segment.dir;
    LineSegment first_segment = line_segments[0];
    glm::vec2 first_vertex = first_segment.v0;

    std::vector<unsigned int> enriched_rects;
    std::vector<EnrichedQuad> enriched_elements;
    std::vector<HeavisideEnrichedQuad> heaviside_enriched_quads;
    std::cout << "Setting level set values for nodes" << std::endl;
    for (size_t i = 0; i < total_vertices; i++)
    {
        // break;
        const glm::vec2 &vertex = vertices[i];
        line_segment_min_dist2 = std::numeric_limits<float>::max();
        for (size_t j = 0; j < line_segments.size(); j++)
        {
            const LineSegment &line_segment = line_segments[j];
            v0_to_vertex = vertex - line_segment.v0;
            if (line_segment.l_squared < 1e-12)
            {
                std::cout << "Degenerate segment" << std::endl;
                continue;
            }
            t = glm::dot(v0_to_vertex, line_segment.dir) / line_segment.l_squared;
            closest_point = line_segment.v0 + glm::clamp(t, 0.0f, 1.0f) * line_segment.dir;
            dist = vertex - closest_point;
            dist2 = glm::dot(dist, dist);
            if (dist2 < line_segment_min_dist2)
            {
                line_segment_min_dist2 = dist2;
                line_segment_min_dist_index = j;
                line_segment_min_dist_t_par = t;
            }
        }
        const LineSegment &closest_line_segment = line_segments[line_segment_min_dist_index];
        v0_to_vertex = vertex - (closest_line_segment.v0 +
                                 glm::clamp(line_segment_min_dist_t_par, 0.0f, 1.0f) * closest_line_segment.dir);
        signed_area = closest_line_segment.dir.x * v0_to_vertex.y - v0_to_vertex.x * closest_line_segment.dir.y;

        sign = 0;
        tip = 0;
        index = 0xdeadbeef;
        t = line_segment_min_dist_t_par;
        if (t > 1.0f)
        {

            if (line_segment_min_dist_index == line_segments.size() - 1)
            {
                index = line_segments.size() - 1;
                sign = signed_area > 0.0f ? 1 : (signed_area < 0.0f ? -1 : 0);
                tip = 1;
            }
            else
            {
                index = line_segment_min_dist_index;
                const glm::vec2 &a = line_segments[line_segment_min_dist_index].dir;
                const glm::vec2 &b = line_segments[line_segment_min_dist_index + 1].dir;

                ab = a.x * b.y - a.y * b.x;
                ac = a.x * v0_to_vertex.y - a.y * v0_to_vertex.x;
                bc = b.x * v0_to_vertex.y - b.y * v0_to_vertex.x;
                sign = ab < 0 ? (ac > 0 || bc > 0 ? 1 : -1) : (ac > 0 && bc > 0 ? 1 : -1);
            }
        }
        else if (t < 0.0f)
        {
            if (line_segment_min_dist_index == 0)
            {
                index = line_segment_min_dist_index;
                sign = signed_area > 0.0f ? 1 : (signed_area < 0.0f ? -1 : 0);
                tip = -1;
            }
            else
            {
                index = line_segment_min_dist_index - 1;
                const glm::vec2 &a = line_segments[line_segment_min_dist_index - 1].dir;
                const glm::vec2 &b = line_segments[line_segment_min_dist_index].dir;

                ab = a.x * b.y - a.y * b.x;
                ac = a.x * v0_to_vertex.y - a.y * v0_to_vertex.x;
                bc = b.x * v0_to_vertex.y - b.y * v0_to_vertex.x;
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
        level_set_1_signed_dist[i] = sign * glm::sqrt(line_segment_min_dist2);
        level_set_2_signed_dist[i] =
            glm::dot(vertex - last_vertex, last_segment.dir) / glm::sqrt(last_segment.l_squared);
        level_set_3_signed_dist[i] =
            -glm::dot(vertex - first_vertex, first_segment.dir) / glm::sqrt(last_segment.l_squared);
        // std::cout << level_set_2_signed_dist[i] << "\n";
        // rectangles.push_back(Rectangle{glm::vec4(vertex, vertex-10.0f/SCR_WIDTH),
        // RainbowColormap::getColorRGBA(glm::sqrt(line_segment_min_dist2+level_set_2_signed_dist[i]*level_set_2_signed_dist[i]))});
        // rectangles.push_back(Rectangle{glm::vec4(vertex, vertex-10.0f/SCR_WIDTH),
        // RainbowColormap::getColorRGBA(glm::abs(glm::atan(level_set_1_signed_dist[i],level_set_2_signed_dist[i]))/2)});
        // rectangles.push_back(Rectangle{glm::vec4(vertex, vertex-10.0f/SCR_WIDTH),
        // RainbowColormap::getColorRGBA(glm::atan(level_set_1_signed_dist[i],level_set_2_signed_dist[i])/1.57)});
        // rectangles.push_back(Rectangle{glm::vec4(vertex, vertex+5.0f/SCR_WIDTH),
        // RainbowColormap::getColorRGBA(std::abs(level_set_1_signed_dist[i]))}); if (sign > 0)
        // {
        //     circles.push_back(Circle{glm::vec3(vertex, 2.0f/SCR_WIDTH), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)});
        // }
        // else if (sign < 0)w
        // {
        //     circles.push_back(Circle{glm::vec3(vertex, 2.0f/SCR_WIDTH), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)});
        // }
        // else
        // {
        //     std::cout << "On line segment or on its continuation if t > 1" << std::endl;
        // }

        vertices_level_set_signs[i].tip = (line_segment_min_dist_index == line_segments.size() - 1 && t > 1.0f)
                                              ? 1
                                              : ((line_segment_min_dist_index == 0 && t < 0.0f) ? -1 : 0);
    }

    std::cout << "searching for enriched elements" << std::endl;

    for (unsigned int j = 0; j < hn - 1; j++)
    {
        for (unsigned int i = 0; i < wn - 1; i++)
        {
            // if (i == 1 and j == 1) continue;
            unsigned int idx = j * wn + i;
            EnrichedQuad enriched_quad;
            enriched_quad.enrichment_type = NoEnrichment;
            EnrichmentType enrichment_type = NoEnrichment;
            // if (idx != 53)
            //     continue;
            if ((vertices_level_set_signs[idx].sign == vertices_level_set_signs[idx + 1].sign &&
                 vertices_level_set_signs[idx + 1].sign == vertices_level_set_signs[idx + wn].sign &&
                 vertices_level_set_signs[idx + wn].sign == vertices_level_set_signs[idx + wn + 1].sign))
            {

                if (vertices_level_set_signs[idx].index != vertices_level_set_signs[idx + 1].index)
                {
                    if (idx + wn < vertices.size() && idx + 1 + wn < vertices.size() &&
                        inQuad(vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
                               crack_vertices[vertices_level_set_signs[idx + 1].index]))
                    {
                        enriched_quad.enrichment_type = Heaviside;
                        enrichment_type = Heaviside;
                    }
                    else if (idx - wn < 0 && idx + 1 - wn < 0 &&
                             inQuad(vertices[idx - wn], vertices[idx - wn + 1], vertices[idx + 1], vertices[idx],
                                    crack_vertices[vertices_level_set_signs[idx + 1].index]))
                    {
                        enriched_quad.enrichment_type = Heaviside;
                        enrichment_type = Heaviside;
                    }
                }
                else if (vertices_level_set_signs[idx + 1].index != vertices_level_set_signs[idx + wn + 1].index)
                {
                    throw std::runtime_error("NotImplemented");
                }
                else if (vertices_level_set_signs[idx + wn + 1].index != vertices_level_set_signs[idx + wn].index)
                {
                    continue;
                    throw std::runtime_error("NotImplemented");
                    // if (idx + wn < vertices.size() && idx + 1 + wn < vertices.size() &&
                    //     !inQuad(vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
                    //             crack_vertices[std::min(vertices_level_set_signs[idx + wn + 1].index,
                    //                                    vertices_level_set_signs[idx + wn].index)]))
                    // {
                    //     continue;
                    // }
                    // enrichment_type = 1;
                }
                else if (vertices_level_set_signs[idx].index != vertices_level_set_signs[idx + wn].index)
                {
                    continue;
                    throw std::runtime_error("NotImplemented");
                    // if (idx + wn < vertices.size() && idx + 1 + wn < vertices.size() &&
                    //     !inQuad(vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
                    //             crack_vertices[std::min(vertices_level_set_signs[idx].index,
                    //                                    vertices_level_set_signs[idx + wn].index)]))
                    // {
                    //     continue;
                    // }
                    // enrichment_type = 1;
                }
            }
            else if ((vertices_level_set_signs[idx].tip == -1 && vertices_level_set_signs[idx + 1].tip == -1 &&
                      vertices_level_set_signs[idx + wn].tip == -1 &&
                      vertices_level_set_signs[idx + wn + 1].tip == -1) ||
                     (vertices_level_set_signs[idx].tip == 1 && vertices_level_set_signs[idx + 1].tip == 1 &&
                      vertices_level_set_signs[idx + wn].tip == 1 && vertices_level_set_signs[idx + wn + 1].tip == 1))
            {
                // no enrichment
            }
            else
            {
                if ((vertices_level_set_signs[idx].tip == -1) + (vertices_level_set_signs[idx + 1].tip == -1) +
                        (vertices_level_set_signs[idx + wn].tip == -1) +
                        (vertices_level_set_signs[idx + wn + 1].tip == -1) >=
                    2)
                {
                    if (inQuad(vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
                                first_vertex))
                    {
                        int intersection_edge = -1;
                        glm::vec2 intersection_point;
                        if (level_set_1_signed_dist[idx] * level_set_1_signed_dist[idx + 1] < 0 &&
                            vertices_level_set_signs[idx].tip == 0 && vertices_level_set_signs[idx + 1].tip == 0)
                        {
                            intersection_edge = 0;
                            intersection_point = interpolate(vertices[idx], vertices[idx + 1], level_set_1_signed_dist[idx],
                                                            level_set_1_signed_dist[idx + 1]);
                        }
                        if (level_set_1_signed_dist[idx + 1] * level_set_1_signed_dist[idx + wn + 1] < 0 &&
                            vertices_level_set_signs[idx + 1].tip == 0 && vertices_level_set_signs[idx + wn + 1].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 1;
                            intersection_point =
                                interpolate(vertices[idx + 1], vertices[idx + wn + 1], level_set_1_signed_dist[idx + 1],
                                            level_set_1_signed_dist[idx + wn + 1]);
                        }
                        if (level_set_1_signed_dist[idx + wn + 1] * level_set_1_signed_dist[idx + wn] < 0 &&
                            vertices_level_set_signs[idx + wn + 1].tip == 0 && vertices_level_set_signs[idx + wn].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 2;
                            intersection_point =
                                interpolate(vertices[idx + wn + 1], vertices[idx + wn],
                                            level_set_1_signed_dist[idx + wn + 1], level_set_1_signed_dist[idx + wn]);
                        }
                        if (level_set_1_signed_dist[idx + wn] * level_set_1_signed_dist[idx] < 0 &&
                            vertices_level_set_signs[idx + wn].tip == 0 && vertices_level_set_signs[idx].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 3;
                            intersection_point =
                                interpolate(vertices[idx + wn], vertices[idx], level_set_1_signed_dist[idx + wn],
                                            level_set_1_signed_dist[idx]);
                        }
                        if (intersection_edge == -1)
                        {
                            throw std::runtime_error("No intersection edge found for tip enrichment");
                        }
                        std::array<unsigned int, 4> indexes = {idx, idx + 1, idx + wn + 1, idx + wn};
                        std::vector<Triangle1> triangulation;
                        triangulation.push_back(
                            Triangle1{first_vertex, vertices[indexes[intersection_edge]], intersection_point});
                        triangulation.push_back(Triangle1{first_vertex, intersection_point,
                                                        vertices[indexes[positive_mod(intersection_edge + 1, 4)]]});
                        for (unsigned int i = 0; i < 3; i++)
                        {
                            triangulation.push_back(
                                Triangle1{first_vertex, vertices[indexes[positive_mod(intersection_edge + i + 1, 4)]],
                                        vertices[indexes[positive_mod(intersection_edge + i + 2, 4)]]});
                        }
                        enriched_quad.enrichment_type = Tip;
                        enriched_quad.triangulation = std::move(triangulation);
                        enrichment_type = Tip;
                    }
                }
                if ((vertices_level_set_signs[idx].tip == 1) + (vertices_level_set_signs[idx + 1].tip == 1) +
                        (vertices_level_set_signs[idx + wn].tip == 1) +
                        (vertices_level_set_signs[idx + wn + 1].tip == 1) >=
                    2)
                {
                    if (inQuad(vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
                                last_vertex))
                    {
                        if (enriched_quad.enrichment_type == Tip)
                            throw std::runtime_error("NotImplemented: two tips in one element");

                        int intersection_edge = -1;
                        glm::vec2 intersection_point;
                        if (level_set_1_signed_dist[idx] * level_set_1_signed_dist[idx + 1] < 0 &&
                            vertices_level_set_signs[idx].tip == 0 && vertices_level_set_signs[idx + 1].tip == 0)
                        {
                            intersection_edge = 0;
                            intersection_point = interpolate(vertices[idx], vertices[idx + 1], level_set_1_signed_dist[idx],
                                                            level_set_1_signed_dist[idx + 1]);
                        }
                        if (level_set_1_signed_dist[idx + 1] * level_set_1_signed_dist[idx + wn + 1] < 0 &&
                            vertices_level_set_signs[idx + 1].tip == 0 && vertices_level_set_signs[idx + wn + 1].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 1;
                            intersection_point =
                                interpolate(vertices[idx + 1], vertices[idx + wn + 1], level_set_1_signed_dist[idx + 1],
                                            level_set_1_signed_dist[idx + wn + 1]);
                        }
                        if (level_set_1_signed_dist[idx + wn + 1] * level_set_1_signed_dist[idx + wn] < 0 &&
                            vertices_level_set_signs[idx + wn + 1].tip == 0 && vertices_level_set_signs[idx + wn].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 2;
                            intersection_point =
                                interpolate(vertices[idx + wn + 1], vertices[idx + wn],
                                            level_set_1_signed_dist[idx + wn + 1], level_set_1_signed_dist[idx + wn]);
                        }
                        if (level_set_1_signed_dist[idx + wn] * level_set_1_signed_dist[idx] < 0 &&
                            vertices_level_set_signs[idx + wn].tip == 0 && vertices_level_set_signs[idx].tip == 0)
                        {
                            if (intersection_edge != -1)
                            {
                                throw std::runtime_error("NotImplemented: more than 1 intersection_edge");
                            }
                            intersection_edge = 3;
                            intersection_point =
                                interpolate(vertices[idx + wn], vertices[idx], level_set_1_signed_dist[idx + wn],
                                            level_set_1_signed_dist[idx]);
                        }
                        if (intersection_edge == -1)
                        {
                            throw std::runtime_error("No intersection edge found for tip enrichment");
                        }
                        std::array<unsigned int, 4> indexes = {idx, idx + 1, idx + wn + 1, idx + wn};

                        std::vector<Triangle1> triangulation;
                        triangulation.push_back(
                            Triangle1{last_vertex, vertices[indexes[intersection_edge]], intersection_point});
                        triangulation.push_back(Triangle1{last_vertex, intersection_point,
                                                        vertices[indexes[positive_mod(intersection_edge + 1, 4)]]});
                        for (unsigned int i = 0; i < 3; i++)
                        {
                            triangulation.push_back(
                                Triangle1{last_vertex, vertices[indexes[positive_mod(intersection_edge + i + 1, 4)]],
                                        vertices[indexes[positive_mod(intersection_edge + i + 2, 4)]]});
                        }
                        enriched_quad.enrichment_type = Tip;
                        enriched_quad.triangulation = std::move(triangulation);
                        enrichment_type = Tip;
                    } 
                }
                if (enrichment_type != Tip)
                {
                    enriched_quad.enrichment_type = Heaviside;
                    enrichment_type = Heaviside;
                    std::array<unsigned int, 4> indexes = {idx, idx + 1, idx + wn + 1, idx + wn};
                    unsigned int intersection_count = 0;
                    std::array<unsigned int, 2> intersection_edges;
                    std::array<glm::vec2, 2> intersection_points;
                    HeavisideEnrichedQuad element;
                    if (level_set_1_signed_dist[idx] * level_set_1_signed_dist[idx + 1] < 0 &&
                        vertices_level_set_signs[idx].tip == 0 && vertices_level_set_signs[idx + 1].tip == 0)
                    {
                        intersection_edges[intersection_count] = 0;
                        intersection_points[intersection_count] =
                            interpolate(vertices[idx], vertices[idx + 1], level_set_1_signed_dist[idx],
                                        level_set_1_signed_dist[idx + 1]);
                        // very easy for linear quad xi=(f1+f2)/(f1-f2) \in [-1, 1]
                        element.intersection_points_local[intersection_count] =
                            glm::vec2{
                                (level_set_1_signed_dist[idx]+level_set_1_signed_dist[idx+1])/(level_set_1_signed_dist[idx]-level_set_1_signed_dist[idx+1]),
                                -1
                            };
                        intersection_count++;
                    }
                    if (level_set_1_signed_dist[idx + 1] * level_set_1_signed_dist[idx + wn + 1] < 0 &&
                        vertices_level_set_signs[idx + 1].tip == 0 && vertices_level_set_signs[idx + wn + 1].tip == 0)
                    {
                        if (intersection_count > 1)
                        {
                            throw std::runtime_error("NotImplemented: more than 2 intersection_edges");
                        }
                        intersection_edges[intersection_count] = 1;
                        intersection_points[intersection_count] =
                            interpolate(vertices[idx + 1], vertices[idx + wn + 1], level_set_1_signed_dist[idx + 1],
                                        level_set_1_signed_dist[idx + wn + 1]);
                        element.intersection_points_local[intersection_count] =
                            glm::vec2{
                                1,
                                (level_set_1_signed_dist[idx + 1]+level_set_1_signed_dist[idx + wn + 1])/(level_set_1_signed_dist[idx + 1]-level_set_1_signed_dist[idx + wn + 1]),
                            };
                        intersection_count++;
                    }
                    if (level_set_1_signed_dist[idx + wn + 1] * level_set_1_signed_dist[idx + wn] < 0 &&
                        vertices_level_set_signs[idx + wn + 1].tip == 0 && vertices_level_set_signs[idx + wn].tip == 0)
                    {
                        if (intersection_count > 1)
                        {
                            throw std::runtime_error("NotImplemented: more than 2 intersection_edges");
                        }
                        intersection_edges[intersection_count] = 2;
                        intersection_points[intersection_count] =
                            interpolate(vertices[idx + wn + 1], vertices[idx + wn],
                                        level_set_1_signed_dist[idx + wn + 1], level_set_1_signed_dist[idx + wn]);
                        element.intersection_points_local[intersection_count] =
                            glm::vec2{
                                -(level_set_1_signed_dist[idx + wn + 1]+level_set_1_signed_dist[idx + wn])/(level_set_1_signed_dist[idx + wn + 1]-level_set_1_signed_dist[idx + wn]),
                                1
                            };
                        intersection_count++;
                    }
                    if (level_set_1_signed_dist[idx + wn] * level_set_1_signed_dist[idx] < 0 &&
                        vertices_level_set_signs[idx + wn].tip == 0 && vertices_level_set_signs[idx].tip == 0)
                    {
                        if (intersection_count > 1)
                        {
                            throw std::runtime_error("NotImplemented: more than 2 intersection_edges");
                        }
                        intersection_edges[intersection_count] = 3;
                        intersection_points[intersection_count] =
                            interpolate(vertices[idx + wn], vertices[idx], level_set_1_signed_dist[idx + wn],
                                        level_set_1_signed_dist[idx]);
                        element.intersection_points_local[intersection_count] =
                            glm::vec2{
                                -1,
                                -(level_set_1_signed_dist[idx + wn] + level_set_1_signed_dist[idx])/(level_set_1_signed_dist[idx + wn]-level_set_1_signed_dist[idx]),
                            };
                        intersection_count++;
                    }
                    std::vector<Triangle1HeavisideSign> partition;
    
                    element.vertex_indices[0] = idx;
                    element.vertex_indices[1] = idx + 1;
                    element.vertex_indices[2] = idx + wn + 1;
                    element.vertex_indices[3] = idx + wn;
                    
                    partition.reserve(4);
                    // std::array<glm::vec2, 5> poly;
                    std::array<unsigned char, 5> poly;
                    unsigned char polyVerticesCount = 0;
                    
                    // poly[0] = vertices[indexes[0]];
                    poly[0] = 0;
                    // since we store in HeavisideEnrichedQuad firstly positive Heaviside triangles
                    // and my code below doesn't know it, we need to somehow rearrange indices
                    // to positive Heaviside triangles should occur first
                    // we check sign by looking at sign of vertex with index indexes[0]
                    // continue;
                    for (unsigned int i = 0; i < 4; i++)
                    {
                        if (intersection_edges[0] == i)
                        {
                            // poly[polyVerticesCount++] = vertices[indexes[i]];
                            // poly[polyVerticesCount++] = intersection_points[0];
                            // poly[polyVerticesCount++] = intersection_points[1];
                            poly[polyVerticesCount++] = i;
                            poly[polyVerticesCount++] = 4;
                            poly[polyVerticesCount++] = 5;
                            i = intersection_edges[1];
                        }
                        else
                        {
                            // poly[polyVerticesCount++] = vertices[indexes[i]];
                            poly[polyVerticesCount++] = i;
                        }
                    }
                    if (polyVerticesCount == 3)
                    {
                        if (vertices_level_set_signs[indexes[0]].sign > 0){
                            element.positive_heaviside_triangles_num = 1;
                            element.triangles[0] = {poly[0], poly[1], poly[2]};
                        }else{
                            // Turned out that first pass is single negative triangle and we write
                            // it to the end. Thus, there will be 3 positive triangles
                            element.positive_heaviside_triangles_num = 3;
                            element.triangles[3] = {poly[0], poly[1], poly[2]};
                        }
                        // partition.push_back(Triangle1HeavisideSign{poly[0], poly[1], poly[2],
                        //                                            vertices_level_set_signs[indexes[0]].sign});
                    }
                    else
                    {
                        if (vertices_level_set_signs[indexes[0]].sign > 0){
                            // There are more than one positive triangles
                            element.positive_heaviside_triangles_num = polyVerticesCount - 2;
                            for (unsigned char i = 1; i < polyVerticesCount - 1; i++)
                            {
                                element.triangles[i-1] = {poly[0], poly[i], poly[static_cast<unsigned char>(i + 1)]};
                            }
                        }else{
                            // 4 - (polyVerticesCount - 2)
                            element.positive_heaviside_triangles_num = 6 - polyVerticesCount;
                            for (unsigned char i = 1; i < polyVerticesCount - 1; i++)
                            {
                                element.triangles[i+element.positive_heaviside_triangles_num-1] = {poly[0], poly[i], poly[static_cast<unsigned char>(i + 1)]};
                            }
                        }
                        // partition.push_back(Triangle1HeavisideSign{poly[0], poly[i], poly[i + 1],
                        //                                         vertices_level_set_signs[indexes[0]].sign});
                    }
                    polyVerticesCount = 0;
                    // poly[polyVerticesCount++] = intersection_points[1];
                    // poly[polyVerticesCount++] = intersection_points[0];
                    poly[polyVerticesCount++] = 5;
                    poly[polyVerticesCount++] = 4;
                    for (unsigned int i = intersection_edges[0] + 1; i < 4; i++)
                    {
                        if (intersection_edges[1] == i)
                        {
                            // poly[polyVerticesCount++] = vertices[indexes[i]];
                            poly[polyVerticesCount++] = i;
                            break;
                        }
                        else
                        {
                            // poly[polyVerticesCount++] = vertices[indexes[i]];
                            poly[polyVerticesCount++] = i;
                        }
                    }
                    if (polyVerticesCount == 3)
                    {
                        if (vertices_level_set_signs[indexes[intersection_edges[1]]].sign > 0){
                            element.triangles[0] = {poly[0], poly[1], poly[2]};
                        }else{
                            element.triangles[3] = {poly[0], poly[1], poly[2]};
                        }
                        // partition.push_back(Triangle1HeavisideSign{
                        //     poly[0], poly[1], poly[2], vertices_level_set_signs[indexes[intersection_edges[1]]].sign});
                    }
                    else
                    {
                        if (vertices_level_set_signs[indexes[intersection_edges[1]]].sign > 0){
                            for (unsigned char i = 1; i < polyVerticesCount - 1; i++)
                            {
                                element.triangles[i-1] = {poly[0], poly[i], poly[static_cast<unsigned char>(i + 1)]};
                            }
                        }else{
                            for (unsigned char i = 1; i < polyVerticesCount - 1; i++)
                            {
                                element.triangles[i+(polyVerticesCount-3)] = {poly[0], poly[i], poly[static_cast<unsigned char>(i + 1)]};
                            }
                        }
                        // partition.push_back(
                        //     Triangle1HeavisideSign{poly[0], poly[i], poly[i + 1],
                        //                            vertices_level_set_signs[indexes[intersection_edges[1]]].sign});
                    }
                    heaviside_enriched_quads.push_back(element);
                    // enriched_quad.triangulation = partition;
                    for (int node : indexes) {
                        heaviside_enriched_nodes[node] = true;
                    }

                }
            }
            if (enrichment_type == NoEnrichment){
                elements.push_back(LinearQuad::Element{idx, idx+1, idx+wn+1, idx+wn});
            }else if(enrichment_type == Tip){
                elements.push_back(LinearQuad::Element{idx, idx+1, idx+wn+1, idx+wn});
            }
            // Quad quad = Quad{vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
            //                  packColor(glm::vec4(0.0f, 1.0f, 0.0f, 0.5f))};

            // if (enriched_quad.enrichment_type == NoEnrichment)
            // {
            //     quads.push_back(Quad{vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
            //                          packColor(glm::vec4(1.0f, 0.0f, 0.0f, 0.5f))});
            //     continue;
            // }
            // else if (enriched_quad.enrichment_type == Heaviside)
            // {
            //     quads.push_back(Quad{vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
            //                          packColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f))});
            // }
            // else if (enriched_quad.enrichment_type == Tip)
            // {
            //     quads.push_back(Quad{vertices[idx], vertices[idx + 1], vertices[idx + wn + 1], vertices[idx + wn],
            //                          packColor(glm::vec4(0.0f, 0.0f, 1.0f, 0.5f))});
            // }
            // enriched_quad.v00 = vertices[idx];
            // enriched_quad.v10 = vertices[idx + 1];
            // enriched_quad.v11 = vertices[idx + wn + 1];
            // enriched_quad.v01 = vertices[idx + wn];
            // enriched_elements.push_back(enriched_quad);
            // std::visit(
            //     [&](auto &vec) {
            //         for (const auto &tri : vec)
            //         {
                //             polygonal_chains.push_back(
            //                 PolygonalChain{std::vector<glm::vec2>{tri.v0, tri.v1, tri.v2, tri.v0},
            //                                glm::vec4{(rand() % 1000) / 1000.0f, (rand() % 1000) / 1000.0f,
            //                                          (rand() % 1000) / 1000.0f, 1.0f}});
            //         }
            //     },
            //     enriched_quad.triangulation);
        }
    }
    const unsigned int num_nodes = vertices.size();
    // heaviside_enriched_quads.clear();
    // heaviside_enriched_nodes.assign(num_nodes, false);
    std::vector<unsigned int> node_offset(num_nodes);
    std::vector<unsigned int> node_ndof(num_nodes);
    unsigned int dof_counter = 0;
    for (unsigned int n = 0; n < num_nodes; ++n) {
        node_offset[n] = dof_counter;
        node_ndof[n] = 2;                        // always two standard DOFs
        if (heaviside_enriched_nodes[n]){
            node_ndof[n] += 2;
        }
        // if (enriched_tip[n])      node_ndof[n] += 4;
        dof_counter += node_ndof[n];
    }
        // After node_offset and node_ndof are ready:
    size_t total_triplets = 0;
    for (const auto& elem : elements) {
        int n_local = 0;
        for (int i = 0; i < 4; ++i) {            // assuming 4 nodes per element
            int node = elem.node_ids[i];
            n_local += node_ndof[node];
        }
        total_triplets += n_local * (n_local + 1) / 2;
    }
    std::vector<Eigen::Triplet<double>> triplets;
    
    glm::vec4 color;
    Eigen::Matrix3d D = setup_D_matrix(2 * std::pow(10, 11), 0.3, true);
    for (const auto& quad: heaviside_enriched_quads){
        Eigen::Matrix<double, 16, 16> Ke;
        
        Ke.setZero();
        std::array<glm::vec2, 6> points = {
            vertices[quad.vertex_indices[0]],
            vertices[quad.vertex_indices[1]],
            vertices[quad.vertex_indices[2]],
            vertices[quad.vertex_indices[3]],
            quad.intersection_points_local[0],
            quad.intersection_points_local[1]
        };
        std::array<glm::vec2, 6> local_points = {
            glm::vec2{-1, -1},
            glm::vec2{1, -1},
            glm::vec2{1, 1},
            glm::vec2{-1, 1},
            quad.intersection_points_local[0],
            quad.intersection_points_local[1]
        };
        int sign = 1;
        const Eigen::Matrix<double, 4, 2> coords{{
            {points[0].x, points[0].y},
            {points[1].x, points[1].y},
            {points[2].x, points[2].y},
            {points[3].x, points[3].y}
        }};
        std::array<int, 4> node_signs;
        for (int n = 0; n < 4; ++n) {
            node_signs[n] = vertices_level_set_signs[quad.vertex_indices[n]].sign;
        }
        // Eigen::Matrix2f J_xy_xieta = shape.dN_xi_eta * coords;
        double total_area = 0.0;
        for (unsigned int i = 0; i < 4; i++){

            const std::array<unsigned char, 3>& triangle = quad.triangles[i];
            if (i >= quad.positive_heaviside_triangles_num){
                sign = -1;
            }
            Eigen::Matrix2d J_xieta_rs{{
                {local_points[triangle[1]].x-local_points[triangle[0]].x, local_points[triangle[2]].x-local_points[triangle[0]].x},
                {local_points[triangle[1]].y-local_points[triangle[0]].y, local_points[triangle[2]].y-local_points[triangle[0]].y}
            }};
            double det_tri = J_xieta_rs.determinant();
        //     std::cout << "Triangle " << i << " vertices (local): "
        //   << local_points[triangle[0]].x << "," << local_points[triangle[0]].y << " ; "
        //   << local_points[triangle[1]].x << "," << local_points[triangle[1]].y << " ; "
        //   << local_points[triangle[2]].x << "," << local_points[triangle[2]].y << "\n";
            for (unsigned int gp = 0; gp < LinearTriangle::TriangleSecondRule::NGauss; gp++){
                float r = LinearTriangle::TriangleSecondRule::gauss_pts[gp][0];
                float s = LinearTriangle::TriangleSecondRule::gauss_pts[gp][1];
                float t = 1 - r - s;
                float xi = local_points[triangle[0]].x*t+
                    local_points[triangle[1]].x*r+
                    local_points[triangle[2]].x*s;
                float eta = local_points[triangle[0]].y*t+
                local_points[triangle[1]].y*r+
                local_points[triangle[2]].y*s;
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
                LinearTriangle::TriangleSecondRule::JacobianData jd;
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
                for (int i = 0; i < 4; ++i) {
                    B(0, 4*i)   = dN_dx_dy(0, i); // du/dx
                    B(0, 4*i+1) = 0;
                    B(1, 4*i) = 0;
                    B(1, 4*i+1) = dN_dx_dy(1, i); // dv/dy
                    B(2, 4*i)   = dN_dx_dy(1, i); // dv/dx
                    B(2, 4*i+1) = dN_dx_dy(0, i); // du/dy

                    double shifted_factor = sign - node_signs[i];   // = 0 on same side, ±2 on opposite
                    B(0, 4*i+2)   = dN_dx_dy(0, i)*shifted_factor; // du/dx
                    B(0, 4*i+3) = 0;
                    B(1, 4*i+2) = 0;
                    B(1, 4*i+3) = dN_dx_dy(1, i)*shifted_factor; // dv/dy
                    B(2, 4*i+2)   = dN_dx_dy(1, i)*shifted_factor; // dv/dx
                    B(2, 4*i+3) = dN_dx_dy(0, i)*shifted_factor; // du/dy
                }
                double factor = LinearTriangle::TriangleSecondRule::gauss_wts[gp] * std::abs(det_tri) * std::abs(jd.detJ);
                if (det_tri < 0) std::cout << "det_tri < 0" << std::endl;
                if (jd.detJ < 0) std::cout << "jd.detJ < 0" << std::endl;
                Ke += factor * (B.transpose() * D * B);
                // total_area += det_tri;
                // Eigen::VectorXd rigid_x(16);
                // rigid_x.setZero();
                // for (int i = 0; i < 4; ++i) rigid_x(4*i) = 1.0;   // standard u_x = 1
                // Eigen::VectorXd strain = B * rigid_x;   // but B is not available after assembly; we need to compute it again or store
                // // Instead, compute the residual Ke * rigid_x
                // Eigen::VectorXd res = Ke * rigid_x;
                // std::cout << "Norm of Ke * rigid_x: " << res.norm() << std::endl;  // should be near 0
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 16, 16>> es(Ke);
        // std::cout << "eigenvalues: " << es.eigenvalues() << std::endl;
        // std::cout << "total_area: " << total_area << std::endl;
        // std::cout << "det: " << Ke.determinant() << std::endl;
        FEMAssemble::addElementSparseUpperStiffness(LinearQuad::Element{quad.vertex_indices[0],
quad.vertex_indices[1],
quad.vertex_indices[2],
quad.vertex_indices[3]}, Ke, triplets, node_offset, node_ndof, 4);
        // std::cout << "enriched " << Ke << std::endl;
        // visualisation
        color = {1.0f, 1.0f, 0.0f, 1.0f};
        for (unsigned int i = 0; i < 4; i++){
            const std::array<unsigned char, 3>& triangle = quad.triangles[i];
            if (i >= quad.positive_heaviside_triangles_num){
                color = {0.0f, 0.0f, 1.0f, 1.0f};
            }
            std::array<glm::vec2, 4> chain;
            for (unsigned int j = 0; j < 3; j++){
                if (triangle[j] < 4){
                    chain[j] = points[triangle[j]];
                }else{
                    float xi = points[triangle[j]].x;
                    float eta = points[triangle[j]].y;
                    float N0 = (1 - xi) * (1 - eta) / 4.0f;
                    float N1 = (1 + xi) * (1 - eta) / 4.0f;
                    float N2 = (1 + xi) * (1 + eta) / 4.0f;
                    float N3 = (1 - xi) * (1 + eta) / 4.0f;
                    chain[j] = N0 * points[0] + N1 * points[1] + N2 * points[2] + N3 * points[3];
                }
            }
            chain[3] = chain[0];
            // vec_chain
            std::vector<glm::vec2> vec_chain;
            vec_chain.reserve(4);
            vec_chain.assign(chain.begin(), chain.end());
            polygonal_chains.push_back(PolygonalChain{vec_chain, color});
        }
    }
    
    std::cout << elements.size() << " " << heaviside_enriched_quads.size() << std::endl;
    std::cout << "Assembling matrix of size: " << dof_counter << std::endl;
    std::cout << "Triplets count: " << total_triplets << std::endl;
    
    Eigen::Matrix<double, 8, 8> Ke;
    Eigen::Matrix<double, 16, 16> Ke_expanded;
    Eigen::Matrix<double, 4, 2> coordMat;
    unsigned int elementsTotal = elements.size();
    unsigned int elementsCreated = 0;

    unsigned int percent = 0, lastPercent = 0;
    for (const LinearQuad::Element &element : elements)
    {
        percent = static_cast<unsigned int>(100.0 * elementsCreated / elementsTotal);
        if (percent > lastPercent)
        { // чтобы не выводить одно и то же значение много раз
            std::cout << percent << "%\n";
            lastPercent = percent;
        }

        for (int i = 0; i < 4; ++i)
        {
            coordMat(i, 0) = vertices[element.node_ids[i]].x;
            coordMat(i, 1) = vertices[element.node_ids[i]].y;
        }
        Eigen::Matrix<double, 3, 2> coordMat1, coordMat2;
        coordMat1.row(0) = coordMat.row(0);
        coordMat1.row(1) = coordMat.row(1);
        coordMat1.row(2) = coordMat.row(2);

        coordMat2.row(0) = coordMat.row(0);
        coordMat2.row(1) = coordMat.row(2);
        coordMat2.row(2) = coordMat.row(3);
        
        Ke = LinearQuad::element_stiffness(coordMat, D, 1);
        // whether no enrichment or blend
        if (heaviside_enriched_nodes[element.node_ids[0]] || heaviside_enriched_nodes[element.node_ids[1]] 
            || heaviside_enriched_nodes[element.node_ids[2]] || heaviside_enriched_nodes[element.node_ids[3]]){
            Ke_expanded.setZero();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Ke_expanded(4*i + 0, 4*j + 0) = Ke(2*i + 0, 2*j + 0);
                    Ke_expanded(4*i + 0, 4*j + 1) = Ke(2*i + 0, 2*j + 1);
                    Ke_expanded(4*i + 1, 4*j + 0) = Ke(2*i + 1, 2*j + 0);
                    Ke_expanded(4*i + 1, 4*j + 1) = Ke(2*i + 1, 2*j + 1);
                }
            }
            // throw std::runtime_error("dsd");
            // std::cout << "expanded\n" << Ke_expanded << std::endl;
            FEMAssemble::addElementSparseUpperStiffness(element, Ke_expanded, triplets, node_offset, node_ndof, 4);
        }else{
            FEMAssemble::addElementSparseUpperStiffness(element, Ke, triplets, node_offset, node_ndof, 2);
            // std::cout << "no enrichment\n" << Ke << std::endl;
        }

        elementsCreated++;
    }
    
    auto K = FEMAssemble::createStiffnessFromTriplets(triplets, dof_counter);

    constexpr bool disable_output = true;

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

    for (unsigned int i = 0; i < wn; i++)
    {
        fixedDofs.push_back(2 * i);
        fixedDofs.push_back(2 * i + 1);
        fixedValues.push_back(0);
        fixedValues.push_back(0);
        // FEMAssemble::fixDOF(K, P, i, UX|UY);
    }
    for (unsigned int i = 0; i < wn; i++)
    {
        int off = node_offset[(wn * (hn - 1) + i)];
        P(off + 1) = 100000;
    }
    // for (unsigned int i = hn * 0.2; i < hn * 0.8; i++){
    //     P((wn*(i)+0)*2) = -10000;
    //     P((wn*(i)+wn-1)*2) = 10000;
    // }
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
    std::cout << "Solving linear system" << std::endl;
    // Eigen::VectorXd u = solveLinearSystemLLT(Eigen::MatrixXd(K), P);
    Eigen::VectorXd u = FEMAssemble::solveSparseSPDUpper(K, P);
    std::cout << "Writing result to file" << std::endl;
    std::ostringstream Uout;
    Uout << u;
    std::ofstream UoutFile("U.txt");
    UoutFile << Uout.str();

    const double constexpr scale = 10000.0;  // factor to make deformation visible
    std::cout << "Scaling results with factor: " << scale << std::endl;
    std::vector<glm::vec2> node_displacement(num_nodes);
    for (int n = 0; n < num_nodes; ++n) {
        int off = node_offset[n];
        // Standard DOFs are the first two at offset 'off' and 'off+1'
        node_displacement[n].x = u(off);
        node_displacement[n].y = u(off+1);
    }
    std::vector<glm::vec2> vertices_displaced = vertices;
    for (int idx = 0; idx < num_nodes; ++idx) {
        vertices_displaced[idx] += glm::vec2(node_displacement[idx].x * scale,
                                            node_displacement[idx].y * scale);
    }

    std::cout << "Creating GUI" << std::endl;
    for (const auto& quad: heaviside_enriched_quads){
        std::array<glm::vec2, 6> points = {
            vertices_displaced[quad.vertex_indices[0]],
            vertices_displaced[quad.vertex_indices[1]],
            vertices_displaced[quad.vertex_indices[2]],
            vertices_displaced[quad.vertex_indices[3]],
            quad.intersection_points_local[0],
            quad.intersection_points_local[1]
        };
        // visualisation
        color = {1.0f, 1.0f, 0.0f, 1.0f};
        for (unsigned int i = 0; i < 4; i++){
            const std::array<unsigned char, 3>& triangle = quad.triangles[i];
            if (i >= quad.positive_heaviside_triangles_num){
                color = {0.0f, 0.0f, 1.0f, 1.0f};
            }
            std::array<glm::vec2, 4> chain;
            for (unsigned int j = 0; j < 3; j++){
                if (triangle[j] < 4){
                    chain[j] = points[triangle[j]];
                }else{
                    float xi = points[triangle[j]].x;
                    float eta = points[triangle[j]].y;
                    float N0 = (1 - xi) * (1 - eta) / 4.0f;
                    float N1 = (1 + xi) * (1 - eta) / 4.0f;
                    float N2 = (1 + xi) * (1 + eta) / 4.0f;
                    float N3 = (1 - xi) * (1 + eta) / 4.0f;
                    chain[j] = N0 * points[0] + N1 * points[1] + N2 * points[2] + N3 * points[3];
                }
            }
            chain[3] = chain[0];
            // vec_chain
            std::vector<glm::vec2> vec_chain;
            vec_chain.reserve(4);
            vec_chain.assign(chain.begin(), chain.end());
            polygonal_chains.push_back(PolygonalChain{vec_chain, color});
        }
    }
    unsigned int idx;
    for (unsigned int i = 0; i < wn - 1; i++)
    {
        for (unsigned int j = 0; j < hn - 1; j++)
        {
            idx = j * wn + i;
            quads.push_back(Quad{vertices_displaced[idx], vertices_displaced[idx + 1], vertices_displaced[idx + wn + 1],
                                 vertices_displaced[idx + wn], packColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f))});
        }
    }

    for (unsigned int i = 0; i < wn; i++)
    {
        for (unsigned int j = 0; j < hn; j++)
        {
            idx = j * wn + i;
            circles.push_back(
                Circle{glm::vec3{vertices_displaced[idx], 1.0f / SCR_WIDTH}, glm::vec4{0.0f, 1.0f, 0.0f, 1.0f}});
        }
    }

    glm::vec2 v1 = vertices_displaced[0], v3 = vertices_displaced[vertices_displaced.size() - 1];
    camera.Position = glm::vec3{(v1 + v3) / 2.0f, 1.0f};
    camera.Zoom = 90.0f;
    camera.MovementSpeed = 0.5f;

    std::vector<Vertex> chain_vertices;
    std::vector<GLint> firstsStrip;
    std::vector<GLsizei> countsStrip;
    std::vector<GLint> firstsLoop;
    std::vector<GLsizei> countsLoop;
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
    Shader chain_program("shaders/pchain.vert", "shaders/pchain.frag");

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
        xfem_shader.setMat4("mvp", MVP);
        xfem_shader.setVec4("color", 1.0f, 0.0f, 0.0f, 1.0f);
        // glBindVertexArray(VAO);
        // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        // glDrawElements(GL_TRIANGLES, indices.size() * 3, GL_UNSIGNED_INT, (void *)0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        if (rectangles.size())
        {
            rectangle_shader.use();
            rectangle_shader.setMat4("mvp", MVP);
            glBindVertexArray(rectangleVAO);
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, rectangles.size());
        }

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
        xfem_shader.use();
        xfem_shader.setMat4("mvp", MVP);
        xfem_shader.setVec4("color", 0.0f, 1.0f, 1.0f, 1.0f);
        glBindVertexArray(lineVAO);
        glDrawElements(GL_LINES, crack_indices.size() * 2, GL_UNSIGNED_INT, (void *)0);

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

        glBindVertexArray(0);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */

        std::cout << 1.0 / deltaTime << std::endl;
        draw = false;
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
