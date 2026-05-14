#include <iostream>

#include <intrin.h>


#include <Eigen/Dense>
#include <glad/glad.h>
#include <glm/glm.hpp>

void checkGLSLVersion() {
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    
    std::cout << "Renderer: " << renderer << std::endl;
    std::cout << "Vendor: " << vendor << std::endl;
    std::cout << "OpenGL Version: " << version << std::endl;
    std::cout << "GLSL Version: " << glslVersion << std::endl;
    
    // Get max supported versions
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "OpenGL Context Version: " << major << "." << minor << std::endl;
    GLint profile;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
    if (profile & GL_CONTEXT_CORE_PROFILE_BIT)
        std::cout << "Core profile" << std::endl;
    else if (profile & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
        std::cout << "Compatibility profile" << std::endl;
}

bool hasAVX2() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuid(cpuInfo, 7);
        return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
    }
    return false;
}
inline glm::vec2 toGlm(const Eigen::Vector2f& v) {
    return glm::vec2(v(0), v(1));
}
inline glm::vec3 toGlm(const Eigen::Vector3f& v) {
    return glm::vec3(v(0), v(1), v(2));
}

// Convert Eigen::Vector4f to glm::vec4
inline glm::vec4 toGlm(const Eigen::Vector4f& v) {
    return glm::vec4(v(0), v(1), v(2), v(3));
}