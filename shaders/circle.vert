#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aCircle_pos_and_radius;
layout (location = 2) in vec4 aCircle_color;

uniform mat4 mvp;

out vec4 Color;

void main()
{
	Color = aCircle_color;
	gl_Position = mvp * vec4(aCircle_pos_and_radius.xy + aPos * aCircle_pos_and_radius.z, 0.0f, 1.0f);
}