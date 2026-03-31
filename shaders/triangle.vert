#version 460 core

layout(location = 0) in vec2 aPos;

layout(std430, binding = 0) readonly buffer ColorBuffer {
    uint triangleColors[];
};

uniform mat4 mvp;

flat out vec4 vColor;

void main() {
    int triIndex = gl_VertexID / 3;
    
    // Renamed from 'packed' to avoid NVIDIA compiler issue
    uint colorPacked = triangleColors[triIndex];
    vColor = unpackUnorm4x8(colorPacked);
    
    gl_Position = mvp * vec4(aPos, 0.0, 1.0);
}