#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aCorners[4]; // location 1, 2, 3, 4
layout (location = 5) in vec4 aColor; // распакован из GL_UNSIGNED_BYTE благодаря GL_TRUE

out vec4 vColor;
out vec2 vPos;

uniform mat4 mvp;

void main()
{
    vColor = aColor;
    vPos = aPos;
	gl_Position = mvp * vec4(aCorners[gl_VertexID % 4], 0.0f, 1.0f);
}