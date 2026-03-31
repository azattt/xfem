#version 430 core
in vec4 vColor;
in vec2 vPos;
out vec4 FragColor;

uniform float borderWidthPx = 0.01;   // desired border width in pixels

void main() {
    // Distance to left/right edges in uv space
    float distLeft   = vPos.x;
    float distRight  = 1.0 - vPos.x;
    // Distance to bottom/top edges
    float distBottom = vPos.y;
    float distTop    = 1.0 - vPos.y;

    // How much uv changes per pixel in x and y directions
    // vec2 fw = vec2(dFdx(vPos.x), dFdy(vPos.y));   // = (dFdx(vPos.x)+dFdy(vPos.x), dFdx(vPos.y)+dFdy(vPos.y))
    vec2 fw = fwidth(vPos);
    // For each edge, the screen‑space distance in uv units is edge_dist / (uv change per pixel)
    // We want border if that screen distance < borderWidthPx.

    float minDistToEdge = 1e10;
    // Left edge
    if (distLeft< borderWidthPx) minDistToEdge = min(minDistToEdge, distLeft);
    // Right edge
    if (distRight < borderWidthPx) minDistToEdge = min(minDistToEdge, distRight);
    // Bottom edge
    if (distBottom< borderWidthPx) minDistToEdge = min(minDistToEdge, distBottom);
    // Top edge
    if (distTop< borderWidthPx) minDistToEdge = min(minDistToEdge, distTop);

    if (minDistToEdge < 1e10) {
        // We are within borderWidthPx pixels of at least one edge
        FragColor = vec4(vColor.xyz * 0.5, vColor.w);
    } else {
        FragColor = vColor;
    }

}