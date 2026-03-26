#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aBbox;
layout (location = 2) in vec4 aColor;

uniform mat4 mvp;

out vec4 Color;

void main()
{
	Color = aColor;
	gl_Position = mvp * vec4(aPos * (aBbox.zw - aBbox.xy) + aBbox.xy, 0.0f, 1.0f);
}