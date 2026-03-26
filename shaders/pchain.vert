#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor; // распакован из GL_UNSIGNED_BYTE благодаря GL_TRUE

out vec4 vColor;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}