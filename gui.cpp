#include "gui.h"

namespace TriangleGUI{
    
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
Renderer& Renderer::instance(){
    static Renderer instance;   // created once, thread-safe
    return instance;
}
void Renderer::addTriangle(const TriangleColored& triangle){
    m_triangles.push_back(triangle);
}
// singleton access
void Renderer::initializeGL(){
    m_shader = std::make_unique<Shader>("shaders/triangle.vert", "shaders/triangle.frag");

    glGenBuffers(1, &m_VBO);
    glGenVertexArrays(1, &m_VAO);
    std::vector<glm::vec2> positions;
    positions.reserve(m_triangles.size() * 3);
    for (const auto& tri : m_triangles) {
        positions.push_back(tri.v0);
        positions.push_back(tri.v1);
        positions.push_back(tri.v2);
    }
    glBindVertexArray(m_VAO);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec2), positions.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void*)0);
    std::vector<uint32_t> colors;
    colors.reserve(m_triangles.size() * 3);
    for (const auto& tri : m_triangles) {
        colors.push_back(tri.color);
    }

    glGenBuffers(1, &m_SSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, colors.size() * sizeof(uint32_t),
                colors.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_SSBO);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindVertexArray(0);
};
void Renderer::draw(const glm::mat4& MVP){
    m_shader->use();
    m_shader->setMat4("mvp", MVP);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, m_triangles.size()*3);
}
    // private:
        // ~Renderer();
        // std::vector<TriangleColored> m_triangles;
        // Shader* m_shader = nullptr;
        // GLuint m_VAO = 0, m_VBO = 0, m_EBO = 0;

}