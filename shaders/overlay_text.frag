#version 420 core
layout(binding = 0) uniform sampler2D textTexture;

in vec2 vUV;
in vec4 vColor;
out vec4 FragColor;

void main()
{
    float alpha = texture(textTexture, vUV).a;
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
