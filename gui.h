#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <memory>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "shader.h"


struct PolygonalChain
{
    std::vector<glm::vec2> points;
    glm::vec4 color;
};
namespace TriangleGUI{
    uint32_t packColor(const glm::vec4 &color);
    struct TriangleColored{
        glm::vec2 v0, v1, v2;
        uint32_t color;
    };
    
     
    // struct QueueBatch{
    //     virtual void draw() = 0;
    //     virtual ~QueueBatch() = default;
    // };
    // std::vector<QueueBatch> queue;
    // void draw(){
    //     for (const QueueBatch& batch: queue){
    //         if (batch.type ==)
    //     }
    // }
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