#include "misc.h"

#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

/*
Needs the active OpenGL context
*/
void checkGLSLVersion() {
    if (!glfwGetCurrentContext()) {
        std::cerr << "checkGLSLVersion: No active OpenGL context!" << std::endl;
        return;
    }
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
