#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <memory>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "shader.h"

namespace TriangleGUI{
    uint32_t packColor(const glm::vec4 &color);
    struct TriangleColored{
        glm::vec2 v0, v1, v2;
        uint32_t color;
    };
    class Renderer {
    public:
        static Renderer& instance(); // singleton access
        void addTriangle(const TriangleColored& triangle);
        void initializeGL();
        void draw(const glm::mat4& MVP);
    private:
        Renderer() = default;
        ~Renderer() = default;
        Renderer(const Renderer&) = delete;
        std::vector<TriangleColored> m_triangles;
        std::vector<glm::vec2> positions;
        std::vector<glm::vec2> colors;
        std::unique_ptr<Shader> m_shader;
        GLuint m_VAO = 0, m_VBO = 0, m_SSBO = 0;
    };
}