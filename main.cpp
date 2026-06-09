#include <algorithm>
#include <array>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <Eigen/Dense>
#include <Eigen/src/SVD/JacobiSVD.h>

#include "calculate.h"
#include "fem.h"
#include "gui.h"
#include "levelset.h"
#include "postprocess.h"

#include "BC.h"

#include "misc.h"

#include "camera.h"
#include "shader.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

class RainbowColormap
{
  public:
    // ANSYS classic rainbow colors (blue -> cyan -> green -> yellow -> red)
    static glm::vec3 getColor(float t)
    {
        // Clamp t to [0, 1] range
        t = glm::clamp(t, 0.0f, 1.0f);

        // ANSYS rainbow typically has 5 segments:
        // 0.0: Blue (0,0,1)
        // 0.25: Cyan (0,1,1)
        // 0.5: Green (0,1,0)
        // 0.75: Yellow (1,1,0)
        // 1.0: Red (1,0,0)

        if (t < 0.25f)
        {
            // Blue to Cyan
            float localT = t / 0.25f;
            return glm::vec3(0.0f,
                             localT, // Green increases
                             1.0f    // Blue constant
            );
        }
        else if (t < 0.5f)
        {
            // Cyan to Green
            float localT = (t - 0.25f) / 0.25f;
            return glm::vec3(0.0f,
                             1.0f,         // Green constant
                             1.0f - localT // Blue decreases
            );
        }
        else if (t < 0.75f)
        {
            // Green to Yellow
            float localT = (t - 0.5f) / 0.25f;
            return glm::vec3(localT, // Red increases
                             1.0f,   // Green constant
                             0.0f);
        }
        else
        {
            // Yellow to Red
            float localT = (t - 0.75f) / 0.25f;
            return glm::vec3(1.0f,          // Red constant
                             1.0f - localT, // Green decreases
                             0.0f);
        }
    }

    // Get color with alpha for transparency
    static glm::vec4 getColorRGBA(float t, float alpha = 1.0f)
    {
        glm::vec3 rgb = getColor(t);
        return glm::vec4(rgb, alpha);
    }
};

struct Triangle
{
    GLuint v0, v1, v2;
};

struct Circle
{
    glm::vec3 center_and_radius;
    glm::vec4 color;
};
struct Rectangle
{
    glm::vec4 bbox;
    glm::vec4 color;
};
struct Quad
{
    glm::vec2 v00, v10, v11, v01;
    uint32_t color;
};

struct Vertex
{
    glm::vec2 position;
    uint32_t colorPacked; // 4 байта: RGBA
};
uint32_t packColor(const glm::vec4 &color)
{
    auto to_uint8 = [](float c) -> uint32_t {
        // Clamp to [0,1] and round to nearest integer
        float clamped = std::clamp(c, 0.0f, 1.0f);
        return static_cast<uint32_t>(clamped * 255.0f + 0.5f) & 0xFF;
    };
    uint32_t r = to_uint8(color.r);
    uint32_t g = to_uint8(color.g);
    uint32_t b = to_uint8(color.b);
    uint32_t a = to_uint8(color.a);

    return (r << 0) | (g << 8) | (b << 16) | (a << 24);
}
// struct PolygonalChainClosed
// {
//     std::vector<glm::vec2> points;
//     glm::vec4 color;
// };

// settings
unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.5f, 0.5f, 2.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

bool lmbPressed = false;
bool rmbPressed = false;

// timing
float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

bool draw = true;

void windowRefreshCallback([[maybe_unused]] GLFWwindow *window)
{
    draw = true;
}
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(UP, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(DOWN, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(LEFT, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(RIGHT, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(FORWARD, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    {
        camera.ProcessKeyboard(BACKWARD, std::min(deltaTime, 1.0f / 60.0f));
        draw = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback([[maybe_unused]] GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    if (width != 0 && height != 0)
    {
        SCR_WIDTH = width;
        SCR_HEIGHT = height;
        glViewport(0, 0, width, height);
        draw = true;
    }
    else
    {
        draw = false;
    }
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback([[maybe_unused]] GLFWwindow *window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    // float xoffset = xpos - lastX;
    // float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    // camera.ProcessMouseMovement(xoffset, yoffset);
}

void mouse_button_callback([[maybe_unused]] GLFWwindow *window, int button, int action, [[maybe_unused]] int mods)
{
    if (button == GLFW_MOUSE_BUTTON_1)
    {
        if (action == GLFW_PRESS)
        {
            lmbPressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            lmbPressed = false;
        }
    }
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback([[maybe_unused]] GLFWwindow *window, [[maybe_unused]] double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void saveCrackToFile(const Crack &crack, const std::string &filename)
{
    std::ofstream out(filename);

    if (!out.is_open())
    {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    if (crack.indices.empty())
    {
        throw std::runtime_error("Cannot save crack: no crack segments");
    }

    // Сохраняем НЕ crack.vertices по порядку хранения,
    // а реальную полилинию по crack.indices.
    out << crack.vertices[crack.indices.front().v0].x() << " " << crack.vertices[crack.indices.front().v0].y() << "\n";

    for (const CrackSegment &seg : crack.indices)
    {
        out << crack.vertices[seg.v1].x() << " " << crack.vertices[seg.v1].y() << "\n";
    }

    std::cout << "Crack saved to: " << filename << std::endl;
}

struct GrowthFrame
{
    Crack crack;
    XFemIterationResult solve_result;
    std::vector<TipKResult> k_results;
};

struct StressTriangleValue
{
    Eigen::Vector2d p0;
    Eigen::Vector2d p1;
    Eigen::Vector2d p2;
    double value = 0.0;
};

struct GrowthFrameVisual
{
    std::vector<Quad> quads;
    std::vector<Circle> circles;
    std::vector<Vertex> crack_vertices;

    // Raw von Mises triangles before coloring.
    // They are converted to von_mises_vertices after a global color range is known.
    std::vector<StressTriangleValue> von_mises_triangles;
    std::vector<Vertex> von_mises_vertices;
};


struct VonMisesScale
{
    double vmin = 0.0;
    double vmax = 1.0;
};

struct OverlayVertex
{
    float x = 0.0f;
    float y = 0.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

struct LegendGL
{
    GLuint vao_tri = 0;
    GLuint vbo_tri = 0;
    GLuint vao_line = 0;
    GLuint vbo_line = 0;
    GLsizei tri_count = 0;
    GLsizei line_count = 0;
};

static glm::vec3 legendColor(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return RainbowColormap::getColor(static_cast<float>(t));
}

static void pushOverlayTriangle(
    std::vector<OverlayVertex>& verts,
    const glm::vec2& p0,
    const glm::vec2& p1,
    const glm::vec2& p2,
    const glm::vec3& c0,
    const glm::vec3& c1,
    const glm::vec3& c2
)
{
    verts.push_back(OverlayVertex{p0.x, p0.y, c0.r, c0.g, c0.b});
    verts.push_back(OverlayVertex{p1.x, p1.y, c1.r, c1.g, c1.b});
    verts.push_back(OverlayVertex{p2.x, p2.y, c2.r, c2.g, c2.b});
}

static void pushOverlayLine(
    std::vector<OverlayVertex>& verts,
    const glm::vec2& p0,
    const glm::vec2& p1,
    const glm::vec3& c
)
{
    verts.push_back(OverlayVertex{p0.x, p0.y, c.r, c.g, c.b});
    verts.push_back(OverlayVertex{p1.x, p1.y, c.r, c.g, c.b});
}

static void buildVonMisesLegendGeometry(
    std::vector<OverlayVertex>& tri_verts,
    std::vector<OverlayVertex>& line_verts
)
{
    tri_verts.clear();
    line_verts.clear();

    // Normalized device coordinates. This stays fixed at the right side of the window.
    const float x0 = 0.70f;
    const float x1 = 0.75f;
    const float y0 = -0.82f;
    const float y1 =  0.82f;

    constexpr int segments = 160;

    for (int i = 0; i < segments; ++i)
    {
        const float t0 = static_cast<float>(i) / static_cast<float>(segments);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(segments);

        const float ya = y0 + (y1 - y0) * t0;
        const float yb = y0 + (y1 - y0) * t1;

        const glm::vec3 c0 = legendColor(t0);
        const glm::vec3 c1 = legendColor(t1);

        const glm::vec2 p00{x0, ya};
        const glm::vec2 p10{x1, ya};
        const glm::vec2 p01{x0, yb};
        const glm::vec2 p11{x1, yb};

        pushOverlayTriangle(tri_verts, p00, p10, p11, c0, c0, c1);
        pushOverlayTriangle(tri_verts, p00, p11, p01, c0, c1, c1);
    }

    const glm::vec3 white{1.0f, 1.0f, 1.0f};

    // Border.
    pushOverlayLine(line_verts, {x0, y0}, {x1, y0}, white);
    pushOverlayLine(line_verts, {x1, y0}, {x1, y1}, white);
    pushOverlayLine(line_verts, {x1, y1}, {x0, y1}, white);
    pushOverlayLine(line_verts, {x0, y1}, {x0, y0}, white);

    // Tick marks.
    constexpr int ticks = 7;
    for (int i = 0; i < ticks; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(ticks - 1);
        const float y = y0 + (y1 - y0) * t;
        pushOverlayLine(line_verts, {x1, y}, {x1 + 0.025f, y}, white);
    }
}

static void initLegendGL(
    LegendGL& legend,
    const std::vector<OverlayVertex>& tri_verts,
    const std::vector<OverlayVertex>& line_verts
)
{
    glGenVertexArrays(1, &legend.vao_tri);
    glGenBuffers(1, &legend.vbo_tri);

    glBindVertexArray(legend.vao_tri);
    glBindBuffer(GL_ARRAY_BUFFER, legend.vbo_tri);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(tri_verts.size() * sizeof(OverlayVertex)),
        tri_verts.empty() ? nullptr : tri_verts.data(),
        GL_STATIC_DRAW
    );

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    legend.tri_count = static_cast<GLsizei>(tri_verts.size());

    glGenVertexArrays(1, &legend.vao_line);
    glGenBuffers(1, &legend.vbo_line);

    glBindVertexArray(legend.vao_line);
    glBindBuffer(GL_ARRAY_BUFFER, legend.vbo_line);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(line_verts.size() * sizeof(OverlayVertex)),
        line_verts.empty() ? nullptr : line_verts.data(),
        GL_STATIC_DRAW
    );

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    legend.line_count = static_cast<GLsizei>(line_verts.size());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

static void drawLegendGL(const LegendGL& legend)
{
    glBindVertexArray(legend.vao_tri);
    glDrawArrays(GL_TRIANGLES, 0, legend.tri_count);

    glBindVertexArray(legend.vao_line);
    glLineWidth(1.5f);
    glDrawArrays(GL_LINES, 0, legend.line_count);

    glBindVertexArray(0);
}


struct TextVertex
{
    float x = 0.0f;
    float y = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
};

struct FontGlyph
{
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 0.0f;
    float v1 = 0.0f;
    int width = 0;
    int height = 0;
    int advance = 0;
};

struct FontAtlasGL
{
    GLuint texture = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    int atlas_width = 0;
    int atlas_height = 0;
    int line_height = 0;
    std::array<FontGlyph, 128> glyphs{};
    GLsizei vertex_count = 0;
};

static glm::vec2 pixelToNdc(float x_px, float y_px, int screen_w, int screen_h)
{
    return glm::vec2{
        2.0f * x_px / static_cast<float>(screen_w) - 1.0f,
        1.0f - 2.0f * y_px / static_cast<float>(screen_h)
    };
}

static float measureTextWidthPx(const FontAtlasGL& font, const std::string& text)
{
    float width = 0.0f;

    for (char c : text)
    {
        const unsigned char uc = static_cast<unsigned char>(c);

        if (uc < font.glyphs.size())
        {
            width += static_cast<float>(font.glyphs[uc].advance);
        }
    }

    return width;
}

static void pushTextQuad(
    std::vector<TextVertex>& out,
    const glm::vec2& p0,
    const glm::vec2& p1,
    const glm::vec2& p2,
    const glm::vec2& p3,
    const FontGlyph& g,
    const glm::vec4& color
)
{
    out.push_back(TextVertex{p0.x, p0.y, g.u0, g.v0, color.r, color.g, color.b, color.a});
    out.push_back(TextVertex{p1.x, p1.y, g.u1, g.v0, color.r, color.g, color.b, color.a});
    out.push_back(TextVertex{p2.x, p2.y, g.u1, g.v1, color.r, color.g, color.b, color.a});

    out.push_back(TextVertex{p0.x, p0.y, g.u0, g.v0, color.r, color.g, color.b, color.a});
    out.push_back(TextVertex{p2.x, p2.y, g.u1, g.v1, color.r, color.g, color.b, color.a});
    out.push_back(TextVertex{p3.x, p3.y, g.u0, g.v1, color.r, color.g, color.b, color.a});
}

static void appendTextPx(
    std::vector<TextVertex>& out,
    const FontAtlasGL& font,
    const std::string& text,
    float x_px,
    float y_px,
    const glm::vec4& color,
    int screen_w,
    int screen_h
)
{
    float pen_x = x_px;

    for (char c : text)
    {
        const unsigned char uc = static_cast<unsigned char>(c);

        if (uc >= font.glyphs.size())
        {
            continue;
        }

        const FontGlyph& g = font.glyphs[uc];

        if (g.width > 0 && g.height > 0)
        {
            const float x0 = pen_x;
            const float x1 = pen_x + static_cast<float>(g.width);
            const float y0 = y_px;
            const float y1 = y_px + static_cast<float>(g.height);

            const glm::vec2 p0 = pixelToNdc(x0, y0, screen_w, screen_h);
            const glm::vec2 p1 = pixelToNdc(x1, y0, screen_w, screen_h);
            const glm::vec2 p2 = pixelToNdc(x1, y1, screen_w, screen_h);
            const glm::vec2 p3 = pixelToNdc(x0, y1, screen_w, screen_h);

            pushTextQuad(out, p0, p1, p2, p3, g, color);
        }

        pen_x += static_cast<float>(g.advance);
    }
}

static std::string formatLegendValue(double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    return std::string(buffer);
}

static std::vector<TextVertex> buildLegendTextVertices(
    const FontAtlasGL& font,
    const VonMisesScale& scale,
    int current_frame,
    int total_frames,
    int screen_w,
    int screen_h
)
{
    std::vector<TextVertex> verts;

    const glm::vec4 white{1.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec4 title_color{0.92f, 0.92f, 0.92f, 1.0f};

    const float bar_x1_ndc = 0.75f;
    const float y0_ndc = -0.82f;
    const float y1_ndc = 0.82f;

    const float label_x_px = (bar_x1_ndc + 1.0f) * 0.5f * static_cast<float>(screen_w) + 12.0f;

    const int ticks = 7;

    for (int i = 0; i < ticks; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(ticks - 1);
        const float y_ndc = y0_ndc + (y1_ndc - y0_ndc) * t;
        const float y_px = (1.0f - y_ndc) * 0.5f * static_cast<float>(screen_h)
                           - static_cast<float>(font.line_height) * 0.5f;

        const double value = scale.vmin + static_cast<double>(t) * (scale.vmax - scale.vmin);
        const std::string label = formatLegendValue(value);

        appendTextPx(
            verts,
            font,
            label,
            label_x_px,
            y_px,
            white,
            screen_w,
            screen_h
        );
    }

    const std::string title = "S: von Mises";
    const float title_width = measureTextWidthPx(font, title);
    const float title_x_px = label_x_px - 3.0f;
    const float title_y_px = (1.0f - y1_ndc) * 0.5f * static_cast<float>(screen_h) - 80.0f;

    appendTextPx(
        verts,
        font,
        title,
        title_x_px,
        title_y_px,
        title_color,
        screen_w,
        screen_h
    );

    char frame_buffer[64];
    std::snprintf(
        frame_buffer,
        sizeof(frame_buffer),
        "Frame %d / %d",
        current_frame + 1,
        total_frames
    );

    appendTextPx(
        verts,
        font,
        frame_buffer,
        title_x_px,
        title_y_px + static_cast<float>(font.line_height) + 4.0f,
        title_color,
        screen_w,
        screen_h
    );

    return verts;
}

#ifdef _WIN32
static FontAtlasGL createWindowsFontAtlasGL(const char* face_name, int pixel_height)
{
    FontAtlasGL font;

    constexpr int atlas_w = 1024;
    constexpr int atlas_h = 512;

    font.atlas_width = atlas_w;
    font.atlas_height = atlas_h;

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = atlas_w;
    bmi.bmiHeader.biHeight = -atlas_h; // top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* dib_bits = nullptr;
    HDC screen_dc = GetDC(nullptr);
    HDC memory_dc = CreateCompatibleDC(screen_dc);

    HBITMAP bitmap = CreateDIBSection(
        memory_dc,
        &bmi,
        DIB_RGB_COLORS,
        &dib_bits,
        nullptr,
        0
    );

    if (!bitmap || !dib_bits)
    {
        throw std::runtime_error("Failed to create text font DIB section");
    }

    HGDIOBJ old_bitmap = SelectObject(memory_dc, bitmap);

    HFONT hfont = CreateFontA(
        -pixel_height,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        ANSI_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_SWISS,
        face_name
    );

    if (!hfont)
    {
        hfont = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    }

    HGDIOBJ old_font = SelectObject(memory_dc, hfont);

    std::memset(dib_bits, 0, atlas_w * atlas_h * 4);

    SetBkMode(memory_dc, TRANSPARENT);
    SetTextColor(memory_dc, RGB(255, 255, 255));

    TEXTMETRICA metrics{};
    GetTextMetricsA(memory_dc, &metrics);
    font.line_height = metrics.tmHeight;

    int pen_x = 4;
    int pen_y = 4;
    const int margin = 3;

    for (int ch = 32; ch < 127; ++ch)
    {
        const char c = static_cast<char>(ch);
        SIZE size{};
        GetTextExtentPoint32A(memory_dc, &c, 1, &size);

        int glyph_w = std::max(1L, size.cx);
        int glyph_h = std::max(1L, metrics.tmHeight);

        if (pen_x + glyph_w + margin >= atlas_w)
        {
            pen_x = 4;
            pen_y += font.line_height + margin;
        }

        if (pen_y + glyph_h + margin >= atlas_h)
        {
            throw std::runtime_error("Font atlas is too small");
        }

        TextOutA(memory_dc, pen_x, pen_y, &c, 1);

        FontGlyph glyph;
        glyph.width = glyph_w;
        glyph.height = glyph_h;
        glyph.advance = std::max((long)glyph_w, (long)size.cx) + 1;
        glyph.u0 = static_cast<float>(pen_x) / static_cast<float>(atlas_w);
        glyph.v0 = static_cast<float>(pen_y) / static_cast<float>(atlas_h);
        glyph.u1 = static_cast<float>(pen_x + glyph_w) / static_cast<float>(atlas_w);
        glyph.v1 = static_cast<float>(pen_y + glyph_h) / static_cast<float>(atlas_h);

        font.glyphs[ch] = glyph;

        pen_x += glyph_w + margin;
    }

    const unsigned char* src = static_cast<const unsigned char*>(dib_bits);
    std::vector<unsigned char> rgba(atlas_w * atlas_h * 4, 0);

    for (int i = 0; i < atlas_w * atlas_h; ++i)
    {
        const unsigned char b = src[4 * i + 0];
        const unsigned char g = src[4 * i + 1];
        const unsigned char r = src[4 * i + 2];
        const unsigned char alpha = std::max(r, std::max(g, b));

        rgba[4 * i + 0] = 255;
        rgba[4 * i + 1] = 255;
        rgba[4 * i + 2] = 255;
        rgba[4 * i + 3] = alpha;
    }

    glGenTextures(1, &font.texture);
    glBindTexture(GL_TEXTURE_2D, font.texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        atlas_w,
        atlas_h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.data()
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenVertexArrays(1, &font.vao);
    glGenBuffers(1, &font.vbo);

    glBindVertexArray(font.vao);
    glBindBuffer(GL_ARRAY_BUFFER, font.vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    SelectObject(memory_dc, old_font);
    DeleteObject(hfont);
    SelectObject(memory_dc, old_bitmap);
    DeleteObject(bitmap);
    DeleteDC(memory_dc);
    ReleaseDC(nullptr, screen_dc);

    return font;
}
#else
static FontAtlasGL createWindowsFontAtlasGL(const char*, int)
{
    throw std::runtime_error("ANSYS-style system font atlas is implemented only for Windows in this patch");
}
#endif

static void uploadTextVertices(FontAtlasGL& font, const std::vector<TextVertex>& vertices)
{
    glBindBuffer(GL_ARRAY_BUFFER, font.vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(TextVertex)),
        vertices.empty() ? nullptr : vertices.data(),
        GL_DYNAMIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    font.vertex_count = static_cast<GLsizei>(vertices.size());
}

static void drawTextGL(const FontAtlasGL& font)
{
    if (font.vertex_count <= 0)
    {
        return;
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, font.texture);
    glBindVertexArray(font.vao);
    glDrawArrays(GL_TRIANGLES, 0, font.vertex_count);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

std::vector<Eigen::Vector2d> makeDisplacedVertices(
    const QuadMesh& mesh,
    const XFemIterationResult& result,
    double scale
)
{
    std::vector<Eigen::Vector2d> vertices_displaced = mesh.vertices;

    for (int n = 0; n < static_cast<int>(mesh.vertices.size()); ++n)
    {
        const int off = static_cast<int>(result.node_offset[n]);

        vertices_displaced[n] +=
            scale * Eigen::Vector2d{result.u(off), result.u(off + 1)};
    }

    return vertices_displaced;
}

std::vector<Vertex> makeCrackLineVertices(
    const Crack& crack,
    const glm::vec4& color
)
{
    std::vector<Vertex> vertices;

    if (crack.indices.empty())
    {
        return vertices;
    }

    const uint32_t packed_color = packColor(color);

    const int first_id = crack.indices.front().v0;
    vertices.push_back(
        Vertex{
            glm::vec2{
                static_cast<float>(crack.vertices[first_id].x()),
                static_cast<float>(crack.vertices[first_id].y())
            },
            packed_color
        }
    );

    for (const CrackSegment& seg : crack.indices)
    {
        const Eigen::Vector2d& p = crack.vertices[seg.v1];

        vertices.push_back(
            Vertex{
                glm::vec2{
                    static_cast<float>(p.x()),
                    static_cast<float>(p.y())
                },
                packed_color
            }
        );
    }

    return vertices;
}


LinearQuad::ShapeData makeQuadShape(double xi, double eta)
{
    LinearQuad::ShapeData shape;

    shape.N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    shape.N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    shape.N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    shape.N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    shape.dN_xi_eta(0, 0) = -0.25 * (1.0 - eta);
    shape.dN_xi_eta(1, 0) = -0.25 * (1.0 - xi);

    shape.dN_xi_eta(0, 1) = 0.25 * (1.0 - eta);
    shape.dN_xi_eta(1, 1) = -0.25 * (1.0 + xi);

    shape.dN_xi_eta(0, 2) = 0.25 * (1.0 + eta);
    shape.dN_xi_eta(1, 2) = 0.25 * (1.0 + xi);

    shape.dN_xi_eta(0, 3) = -0.25 * (1.0 + eta);
    shape.dN_xi_eta(1, 3) = 0.25 * (1.0 - xi);

    return shape;
}

double vonMisesPlaneStress(const Eigen::Vector3d& stress)
{
    const double sx = stress(0);
    const double sy = stress(1);
    const double txy = stress(2);

    return std::sqrt(
        sx * sx - sx * sy + sy * sy + 3.0 * txy * txy
    );
}

glm::vec4 vonMisesColor(double value, double vmin, double vmax)
{
    double t = 0.0;

    if (vmax > vmin)
    {
        t = (value - vmin) / (vmax - vmin);
    }

    t = std::clamp(t, 0.0, 1.0);

    return RainbowColormap::getColorRGBA(
        static_cast<float>(t),
        0.88f
    );
}

struct TipContext
{
    bool valid = false;
    Eigen::Vector2d tip_point = Eigen::Vector2d::Zero();
    Eigen::Vector2d t_vec = Eigen::Vector2d::UnitX();
    Eigen::Vector2d n_vec = Eigen::Vector2d::UnitY();
    std::array<std::array<double, 4>, 4> f_nodes{};
};

TipContext makeTipContext(
    const TipEnriched& tip_data,
    const std::array<int, 4>& element,
    const QuadMesh& mesh,
    const XFemIterationResult& result
)
{
    TipContext ctx;
    ctx.valid = true;

    Eigen::Matrix<double, 4, 2> coords;

    for (int n = 0; n < 4; ++n)
    {
        const Eigen::Vector2d& p = mesh.vertices[element[n]];
        coords(n, 0) = p.x();
        coords(n, 1) = p.y();
    }

    const LinearQuad::ShapeData tip_shape =
        makeQuadShape(
            tip_data.tip_point_local_coords.x(),
            tip_data.tip_point_local_coords.y()
        );

    ctx.tip_point.setZero();

    for (int n = 0; n < 4; ++n)
    {
        ctx.tip_point +=
            tip_shape.N[n] * coords.row(n).transpose();
    }

    if (tip_data.tip_index == 1)
    {
        ctx.t_vec = result.crack_tip_1_t.normalized();
        ctx.n_vec = result.crack_tip_1_n.normalized();
    }
    else
    {
        ctx.t_vec = result.crack_tip_2_t.normalized();
        ctx.n_vec = result.crack_tip_2_n.normalized();
    }

    for (int n = 0; n < 4; ++n)
    {
        const Eigen::Vector2d d =
            coords.row(n).transpose() - ctx.tip_point;

        const double x1 = d.dot(ctx.t_vec);
        const double x2 = d.dot(ctx.n_vec);
        const double r = std::sqrt(x1 * x1 + x2 * x2);

        if (r < 1e-30)
        {
            ctx.f_nodes[n] = {0.0, 0.0, 0.0, 0.0};
            continue;
        }

        const double theta = std::atan2(x2, x1);
        const double sqrt_r = std::sqrt(r);
        const double sin_half = std::sin(theta / 2.0);
        const double cos_half = std::cos(theta / 2.0);
        const double sin_theta = std::sin(theta);

        ctx.f_nodes[n] = {
            sqrt_r * sin_half,
            sqrt_r * cos_half,
            sqrt_r * sin_theta * sin_half,
            sqrt_r * sin_theta * cos_half
        };
    }

    return ctx;
}

struct FieldEvalResult
{
    Eigen::Vector2d x_deformed = Eigen::Vector2d::Zero();
    Eigen::Vector3d stress = Eigen::Vector3d::Zero();
    double von_mises = 0.0;
};

FieldEvalResult evaluateVonMisesPoint(
    const std::array<int, 4>& element,
    const QuadMesh& mesh,
    const XFemIterationResult& result,
    const Eigen::Matrix3d& D,
    double scale,
    double xi,
    double eta,
    int forced_H_value,
    bool use_heaviside,
    bool use_tip,
    const TipContext* tip_ctx
)
{
    FieldEvalResult out;

    Eigen::Matrix<double, 4, 2> coords;

    for (int n = 0; n < 4; ++n)
    {
        const Eigen::Vector2d& p = mesh.vertices[element[n]];
        coords(n, 0) = p.x();
        coords(n, 1) = p.y();
    }

    const LinearQuad::ShapeData shape =
        makeQuadShape(xi, eta);

    LinearTriangle::JacobianData jd;
    jd.J = shape.dN_xi_eta * coords;

    bool invertible = false;
    jd.J.computeInverseAndDetWithCheck(
        jd.invJ,
        jd.detJ,
        invertible,
        1e-12
    );

    if (!invertible)
    {
        throw std::runtime_error(
            "evaluateVonMisesPoint: Jacobi matrix is not invertible"
        );
    }

    const Eigen::Matrix<double, 2, 4> dN_dx_dy =
        jd.invJ * shape.dN_xi_eta;

    Eigen::Vector2d x_global = Eigen::Vector2d::Zero();

    for (int n = 0; n < 4; ++n)
    {
        x_global += shape.N[n] * coords.row(n).transpose();
    }

    Eigen::Vector2d u_global = Eigen::Vector2d::Zero();
    Eigen::Matrix2d grad_u = Eigen::Matrix2d::Zero();

    for (int n = 0; n < 4; ++n)
    {
        const int node = element[n];
        const int off = static_cast<int>(result.node_offset[node]);

        const double Nn = shape.N[n];
        const double dNdx = dN_dx_dy(0, n);
        const double dNdy = dN_dx_dy(1, n);

        const double ux = result.u(off);
        const double uy = result.u(off + 1);

        u_global += Nn * Eigen::Vector2d{ux, uy};

        grad_u(0, 0) += dNdx * ux;
        grad_u(0, 1) += dNdy * ux;
        grad_u(1, 0) += dNdx * uy;
        grad_u(1, 1) += dNdy * uy;

        if (use_heaviside &&
            node < static_cast<int>(result.enriched_elements.heaviside_enriched_nodes.size()) &&
            result.enriched_elements.heaviside_enriched_nodes[node])
        {
            const int H_i =
                result.level_set_fields.vertices_level_set_signs[node].sign;

            const double H_shift =
                static_cast<double>(forced_H_value - H_i);

            const double ax = result.u(off + 2);
            const double ay = result.u(off + 3);

            u_global +=
                Nn * H_shift * Eigen::Vector2d{ax, ay};

            grad_u(0, 0) += dNdx * H_shift * ax;
            grad_u(0, 1) += dNdy * H_shift * ax;
            grad_u(1, 0) += dNdx * H_shift * ay;
            grad_u(1, 1) += dNdy * H_shift * ay;
        }
    }

    if (use_tip && tip_ctx != nullptr && tip_ctx->valid)
    {
        const Eigen::Vector2d d =
            x_global - tip_ctx->tip_point;

        const double x1 = d.dot(tip_ctx->t_vec);
        const double x2 = d.dot(tip_ctx->n_vec);

        const double r2 = x1 * x1 + x2 * x2;
        const double r = std::sqrt(r2);

        if (r > 1e-14)
        {
            const double theta = std::atan2(x2, x1);
            const double sqrt_r = std::sqrt(r);
            const double inv_sqrt_r = 1.0 / sqrt_r;

            const double sin_half = std::sin(theta / 2.0);
            const double cos_half = std::cos(theta / 2.0);
            const double sin_theta = std::sin(theta);
            const double cos_theta = std::cos(theta);

            std::array<double, 4> f = {
                sqrt_r * sin_half,
                sqrt_r * cos_half,
                sqrt_r * sin_theta * sin_half,
                sqrt_r * sin_theta * cos_half
            };

            std::array<double, 4> df_dr;
            std::array<double, 4> df_dtheta;

            df_dr[0] = 0.5 * inv_sqrt_r * sin_half;
            df_dr[1] = 0.5 * inv_sqrt_r * cos_half;
            df_dr[2] = 0.5 * inv_sqrt_r * sin_half * sin_theta;
            df_dr[3] = 0.5 * inv_sqrt_r * cos_half * sin_theta;

            df_dtheta[0] = sqrt_r * 0.5 * cos_half;
            df_dtheta[1] = -sqrt_r * 0.5 * sin_half;
            df_dtheta[2] =
                sqrt_r *
                (0.5 * cos_half * sin_theta +
                 sin_half * cos_theta);
            df_dtheta[3] =
                sqrt_r *
                (-0.5 * sin_half * sin_theta +
                 cos_half * cos_theta);

            const double drdx = d.x() / r;
            const double drdy = d.y() / r;

            const double dtheta_dx =
                (x1 * tip_ctx->n_vec.x() -
                 x2 * tip_ctx->t_vec.x()) / r2;

            const double dtheta_dy =
                (x1 * tip_ctx->n_vec.y() -
                 x2 * tip_ctx->t_vec.y()) / r2;

            std::array<Eigen::Vector2d, 4> df_dx;

            for (int a = 0; a < 4; ++a)
            {
                df_dx[a].x() =
                    df_dr[a] * drdx +
                    df_dtheta[a] * dtheta_dx;

                df_dx[a].y() =
                    df_dr[a] * drdy +
                    df_dtheta[a] * dtheta_dy;
            }

            for (int n = 0; n < 4; ++n)
            {
                const int node = element[n];

                if (node >= static_cast<int>(result.enriched_elements.tip_enriched_nodes.size()) ||
                    !result.enriched_elements.tip_enriched_nodes[node])
                {
                    continue;
                }

                const int off = static_cast<int>(result.node_offset[node]);
                const double Nn = shape.N[n];
                const double dNdx = dN_dx_dy(0, n);
                const double dNdy = dN_dx_dy(1, n);

                for (int a = 0; a < 4; ++a)
                {
                    const double bx = result.u(off + 4 + 2 * a);
                    const double by = result.u(off + 4 + 2 * a + 1);

                    const double shift =
                        f[a] - tip_ctx->f_nodes[n][a];

                    const double d_enr_dx =
                        dNdx * shift + Nn * df_dx[a].x();

                    const double d_enr_dy =
                        dNdy * shift + Nn * df_dx[a].y();

                    u_global +=
                        Nn * shift * Eigen::Vector2d{bx, by};

                    grad_u(0, 0) += d_enr_dx * bx;
                    grad_u(0, 1) += d_enr_dy * bx;
                    grad_u(1, 0) += d_enr_dx * by;
                    grad_u(1, 1) += d_enr_dy * by;
                }
            }
        }
    }

    Eigen::Vector3d strain;
    strain << grad_u(0, 0),
              grad_u(1, 1),
              grad_u(0, 1) + grad_u(1, 0);

    out.stress = D * strain;
    out.von_mises = vonMisesPlaneStress(out.stress);
    out.x_deformed = x_global + scale * u_global;

    return out;
}

void addVonMisesTriangle(
    std::vector<StressTriangleValue>& output,
    const std::array<int, 4>& element,
    const QuadMesh& mesh,
    const XFemIterationResult& result,
    const Eigen::Matrix3d& D,
    double scale,
    const std::array<Eigen::Vector2d, 3>& local_tri,
    int forced_H_value,
    bool use_heaviside,
    bool use_tip,
    const TipContext* tip_ctx
)
{
    const double xi_c =
        (local_tri[0].x() + local_tri[1].x() + local_tri[2].x()) / 3.0;

    const double eta_c =
        (local_tri[0].y() + local_tri[1].y() + local_tri[2].y()) / 3.0;

    const FieldEvalResult c =
        evaluateVonMisesPoint(
            element,
            mesh,
            result,
            D,
            scale,
            xi_c,
            eta_c,
            forced_H_value,
            use_heaviside,
            use_tip,
            tip_ctx
        );

    const FieldEvalResult v0 =
        evaluateVonMisesPoint(
            element,
            mesh,
            result,
            D,
            scale,
            local_tri[0].x(),
            local_tri[0].y(),
            forced_H_value,
            use_heaviside,
            use_tip,
            tip_ctx
        );

    const FieldEvalResult v1 =
        evaluateVonMisesPoint(
            element,
            mesh,
            result,
            D,
            scale,
            local_tri[1].x(),
            local_tri[1].y(),
            forced_H_value,
            use_heaviside,
            use_tip,
            tip_ctx
        );

    const FieldEvalResult v2 =
        evaluateVonMisesPoint(
            element,
            mesh,
            result,
            D,
            scale,
            local_tri[2].x(),
            local_tri[2].y(),
            forced_H_value,
            use_heaviside,
            use_tip,
            tip_ctx
        );

    StressTriangleValue tri;
    tri.p0 = v0.x_deformed;
    tri.p1 = v1.x_deformed;
    tri.p2 = v2.x_deformed;
    tri.value = c.von_mises;

    output.push_back(tri);
}

std::vector<StressTriangleValue> buildVonMisesTriangles(
    const GrowthFrame& frame,
    const QuadMesh& mesh,
    const Eigen::Matrix3d& D,
    double scale
)
{
    std::vector<StressTriangleValue> triangles;

    const XFemIterationResult& result = frame.solve_result;

    std::vector<int> tip_element_to_index(
        mesh.elements.size(),
        -1
    );

    for (int i = 0; i < static_cast<int>(result.enriched_elements.tip_enriched.size()); ++i)
    {
        tip_element_to_index[result.enriched_elements.tip_enriched[i].id] = i;
    }

    std::vector<int> heaviside_element_to_index(
        mesh.elements.size(),
        -1
    );

    for (int i = 0; i < static_cast<int>(result.enriched_elements.heaviside_enriched.size()); ++i)
    {
        heaviside_element_to_index[result.enriched_elements.heaviside_enriched[i].id] = i;
    }

    for (int elem_id = 0; elem_id < static_cast<int>(mesh.elements.size()); ++elem_id)
    {
        const std::array<int, 4>& element = mesh.elements[elem_id];
        const int tip_index = tip_element_to_index[elem_id];
        const int h_index = heaviside_element_to_index[elem_id];

        if (tip_index >= 0)
        {
            const TipEnriched& tip_data =
                result.enriched_elements.tip_enriched[tip_index];

            const TipTriangulation& triangulation =
                result.enriched_elements_triangulation.tip_enriched_triangulation[tip_index];

            const TipContext tip_ctx =
                makeTipContext(
                    tip_data,
                    element,
                    mesh,
                    result
                );

            const std::array<Eigen::Vector2d, 6> local_points = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0},
                tip_data.intersection_point_local_coords,
                tip_data.tip_point_local_coords
            };

            for (unsigned int tri_id = 0; tri_id < 5; ++tri_id)
            {
                const std::array<unsigned char, 3>& tri =
                    triangulation.tri_indices[tri_id];

                const std::array<Eigen::Vector2d, 3> local_tri = {
                    local_points[tri[0]],
                    local_points[tri[1]],
                    local_points[tri[2]]
                };

                addVonMisesTriangle(
                    triangles,
                    element,
                    mesh,
                    result,
                    D,
                    scale,
                    local_tri,
                    0,
                    false,
                    true,
                    &tip_ctx
                );
            }
        }
        else if (h_index >= 0)
        {
            const HeavisideEnriched& h_data =
                result.enriched_elements.heaviside_enriched[h_index];

            const HeavisideTriangulation& triangulation =
                result.enriched_elements_triangulation.heaviside_enriched_triangulation[h_index];

            const std::array<Eigen::Vector2d, 6> local_points = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0},
                h_data.intersection_points_local_coords[0],
                h_data.intersection_points_local_coords[1]
            };

            for (unsigned int tri_id = 0; tri_id < triangulation.triangles_num; ++tri_id)
            {
                const std::array<unsigned char, 3>& tri =
                    triangulation.tri_indices[tri_id];

                const std::array<Eigen::Vector2d, 3> local_tri = {
                    local_points[tri[0]],
                    local_points[tri[1]],
                    local_points[tri[2]]
                };

                const int H_value =
                    tri_id < triangulation.positive_heaviside_triangles_num
                        ? +1
                        : -1;

                addVonMisesTriangle(
                    triangles,
                    element,
                    mesh,
                    result,
                    D,
                    scale,
                    local_tri,
                    H_value,
                    true,
                    false,
                    nullptr
                );
            }
        }
        else
        {
            const std::array<Eigen::Vector2d, 3> tri_1 = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0}
            };

            const std::array<Eigen::Vector2d, 3> tri_2 = {
                Eigen::Vector2d{-1.0, -1.0},
                Eigen::Vector2d{ 1.0,  1.0},
                Eigen::Vector2d{-1.0,  1.0}
            };

            addVonMisesTriangle(
                triangles,
                element,
                mesh,
                result,
                D,
                scale,
                tri_1,
                0,
                false,
                false,
                nullptr
            );

            addVonMisesTriangle(
                triangles,
                element,
                mesh,
                result,
                D,
                scale,
                tri_2,
                0,
                false,
                false,
                nullptr
            );
        }
    }

    return triangles;
}

VonMisesScale applyVonMisesColors(
    std::vector<GrowthFrameVisual>& visuals
)
{
    std::vector<double> all_values;

    for (const GrowthFrameVisual& visual : visuals)
    {
        for (const StressTriangleValue& tri : visual.von_mises_triangles)
        {
            if (std::isfinite(tri.value))
            {
                all_values.push_back(tri.value);
            }
        }
    }

    if (all_values.empty())
    {
        return VonMisesScale{};
    }

    std::sort(all_values.begin(), all_values.end());

    const std::size_t low_index =
        static_cast<std::size_t>(0.02 * static_cast<double>(all_values.size() - 1));

    const std::size_t high_index =
        static_cast<std::size_t>(0.98 * static_cast<double>(all_values.size() - 1));

    double vmin = all_values[low_index];
    double vmax = all_values[high_index];

    if (!(vmax > vmin))
    {
        vmin = all_values.front();
        vmax = all_values.back();
    }

    if (!(vmax > vmin))
    {
        vmax = vmin + 1.0;
    }

    std::cout << "Global von Mises color range, clipped 2%..98%: "
              << vmin << " ... " << vmax << std::endl;

    for (GrowthFrameVisual& visual : visuals)
    {
        visual.von_mises_vertices.clear();
        visual.von_mises_vertices.reserve(
            visual.von_mises_triangles.size() * 3
        );

        for (const StressTriangleValue& tri : visual.von_mises_triangles)
        {
            const uint32_t packed_color =
                packColor(vonMisesColor(tri.value, vmin, vmax));

            visual.von_mises_vertices.push_back(
                Vertex{
                    toGlm(tri.p0.cast<float>().eval()),
                    packed_color
                }
            );

            visual.von_mises_vertices.push_back(
                Vertex{
                    toGlm(tri.p1.cast<float>().eval()),
                    packed_color
                }
            );

            visual.von_mises_vertices.push_back(
                Vertex{
                    toGlm(tri.p2.cast<float>().eval()),
                    packed_color
                }
            );
        }
    }

    return VonMisesScale{vmin, vmax};
}

GrowthFrameVisual makeGrowthFrameVisual(
    const GrowthFrame& frame,
    const QuadMesh& mesh,
    const Eigen::Matrix3d& D,
    double scale
)
{
    GrowthFrameVisual visual;

    const std::vector<Eigen::Vector2d> vertices_displaced =
        makeDisplacedVertices(
            mesh,
            frame.solve_result,
            scale
        );

    visual.circles.reserve(mesh.vertices.size());

    for (int idx = 0; idx < static_cast<int>(mesh.vertices.size()); ++idx)
    {
        visual.circles.push_back(
            Circle{
                glm::vec3{
                    toGlm(vertices_displaced[idx].cast<float>().eval()),
                    1.0f / static_cast<float>(SCR_WIDTH)
                },
                glm::vec4{0.0f, 1.0f, 0.0f, 1.0f}
            }
        );
    }

    visual.crack_vertices =
        makeCrackLineVertices(
            frame.crack,
            glm::vec4(0.0f, 1.0f, 1.0f, 1.0f)
        );

    visual.von_mises_triangles =
        buildVonMisesTriangles(
            frame,
            mesh,
            D,
            scale
        );

    return visual;
}

void uploadGrowthFrameVisual(
    const GrowthFrameVisual& visual,
    GLuint quadVBO,
    GLuint circleVBO,
    GLuint chainVBO,
    GLuint vonMisesVBO
)
{
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        visual.quads.size() * sizeof(Quad),
        visual.quads.empty() ? nullptr : visual.quads.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        visual.circles.size() * sizeof(Circle),
        visual.circles.empty() ? nullptr : visual.circles.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, chainVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        visual.crack_vertices.size() * sizeof(Vertex),
        visual.crack_vertices.empty() ? nullptr : visual.crack_vertices.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, vonMisesVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        visual.von_mises_vertices.size() * sizeof(Vertex),
        visual.von_mises_vertices.empty() ? nullptr : visual.von_mises_vertices.data(),
        GL_DYNAMIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool keyPressedOnce(
    GLFWwindow* window,
    int key,
    std::unordered_set<int>& pressed_keys
)
{
    const int state = glfwGetKey(window, key);

    if (state == GLFW_PRESS)
    {
        if (!pressed_keys.contains(key))
        {
            pressed_keys.insert(key);
            return true;
        }
    }
    else
    {
        pressed_keys.erase(key);
    }

    return false;
}

int main()
{
    std::cout << "CPU has AVX2: " << hasAVX2() << std::endl;
    std::ifstream mesh_data("mesh/mesh.txt");
    if (!mesh_data.is_open())
    {
        throw std::runtime_error("Couldn't open mesh/mesh.txt");
    }
    double w, h;
    int wn, hn;
    double thickness; // thickness
    double scale = 0; // factor to make deformation visible

    mesh_data >> w >> h >> wn >> hn >> thickness >> scale;
    // scale = 0;
    // double w = 1, h = 1;
    // int wn = 2, hn = 3;
    double wh = w / (wn - 1), hh = h / (hn - 1);
    std::vector<LinearQuad::Element> elements;

    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    // seed = 1773229899;
    std::cout << "seed: " << seed << std::endl;
    std::srand(seed);

    // for (int i = 0; i < 5; i++){
    //     crack_vertices.push_back(glm::vec2{0.2f-0.1f*glm::cos(2*glm::pi<float>()/5*i),
    //     0.1f*glm::sin(2*glm::pi<float>()/5*i) + 0.5f}); crack_indices.push_back(Line{static_cast<unsigned
    //     int>(crack_vertices.size() - 2), static_cast<unsigned int>(crack_vertices.size() - 1)});
    // }
    // crack_vertices.push_back(glm::vec2{0.1f, 0.5f});
    // for (int i = 2; i < 10; i++)
    // {
    //     // crack_vertices.push_back(glm::vec2{(float)(rand() % 100) / 100, (float)(rand() % 100) / 100});
    //     // crack_vertices.push_back(glm::vec2{(float)i / 10, glm::sin((float)2 * i / 5) / 4 + 0.5f});
    //     // crack_vertices.push_back(glm::vec2{(float)i / 10, (float)(rand() % 100) / 1000 + 0.5f});
    //     crack_vertices.push_back(glm::vec2{(float)i / 10, 0.5f});
    //     // crack_vertices.push_back(glm::vec2{glm::cos((float)2 * i / 5) / 4 + 0.5f, glm::sin((float)2 * i / 5) / 4 +
    //     // 0.5f});
    //     crack_indices.push_back(Line{static_cast<unsigned int>(crack_vertices.size() - 2),
    //                                  static_cast<unsigned int>(crack_vertices.size() - 1)});
    // }

    // crack_vertices.push_back(glm::vec2{0.0f, 0.5f});
    // crack_vertices.push_back(glm::vec2{0.5f, 0.5f});
    // crack_indices.push_back(Line{0, 1});

    // through node
    // crack_vertices.push_back(glm::vec2{0.6f, 0.2f});
    // crack_vertices.push_back(glm::vec2{0.9f, 0.3f});

    int total_vertices = wn * hn;
    QuadMesh mesh;
    mesh.vertices = std::vector<Eigen::Vector<double, 2>>();
    mesh.vertices.reserve(total_vertices);

    for (int j = 0; j < hn; j++)
    {
        for (int i = 0; i < wn; i++)
        {
            mesh.vertices.push_back(Eigen::Vector<double, 2>{i * wh, j * hh});
            // mesh.vertices.push_back(Eigen::Vector<double, 2>{i * wh + (rand() % 100) / 1000.0, j * hh + (rand() %
            // 100) / 1000.0});
        }
    }

    mesh.elements = std::vector<std::array<int, 4>>();
    mesh.elements.reserve((hn - 1) * (wn - 1));
    for (int j = 0; j < hn - 1; j++)
    {
        for (int i = 0; i < wn - 1; i++)
        {
            if ((i >= wn / 6 - 1 && i <= 2 * wn / 6) && j == 3 * hn / 5)
                continue;
            mesh.elements.push_back(
                std::array<int, 4>{j * wn + i, j * wn + i + 1, (j + 1) * wn + i + 1, (j + 1) * wn + i});
        }
    }

    Crack crack;

    std::ifstream crack_data("mesh/crack.txt");
    if (!crack_data.is_open())
    {
        throw std::runtime_error("Couldn't open mesh/crack.txt");
    }
    double a, b;
    while (crack_data >> a >> b)
    {
        crack.vertices.push_back(Eigen::Vector2d(a, b));
    }
    if (crack.vertices.size() < 2)
    {
        throw std::runtime_error("Crack must contain at least two points");
    }
    for (int i = 1; i < crack.vertices.size(); i++)
    {
        crack.indices.push_back(CrackSegment{i - 1, i});
    }

    std::ifstream interaction_integral_data("mesh/interaction_integral.txt");
    if (!interaction_integral_data.is_open())
    {
        throw std::runtime_error("Couldn't open mesh/interaction_integral.txt");
    }
    double Rin, Rout;
    interaction_integral_data >> Rin >> Rout;

    bool disable_output = true;
    std::ifstream disable_output_data("mesh/disable_output.txt");
    if (!disable_output_data.is_open())
    {
        std::cerr << "Couldn't open mesh/disable_output.txt. disable_output set to true." << std::endl;
    }
    else
    {
        disable_output_data >> disable_output;
    }
    bool disable_debug_output = true;
    std::ifstream disable_debug_output_data("mesh/disable_debug_output.txt");
    if (!disable_debug_output_data.is_open())
    {
        std::cerr << "Couldn't open mesh/disable_debug_output.txt. disable_debug_output set to true." << std::endl;
    }
    else
    {
        disable_debug_output_data >> disable_debug_output;
    }
    const double E = 70e9;        // Pa
    constexpr double nu = 0.23;   // -
    const double KIC = 0.75e6;    // Pa * sqrt(m)
    const Eigen::Matrix3d D = setup_D_matrix(E, nu, true);

    const int max_growth_steps = 100;
    std::vector<GrowthFrame> growth_frames;
    growth_frames.reserve(max_growth_steps);

    const double da = 1.0 * std::min(wh, hh);

    XFemIterationResult solve_result;

    for (int growth_step = 0; growth_step < max_growth_steps; ++growth_step)
    {
        std::cout << "\n==============================\n";
        std::cout << "CRACK GROWTH ITERATION " << growth_step << "\n";
        std::cout << "==============================\n";

        solve_result =
            solveCrackIteration(
                mesh,
                crack,
                w,
                h,
                wn,
                hn,
                thickness,
                D,
                disable_output,
                disable_debug_output
            );

        std::vector<TipKResult> k_results =
            computeStress<13>(
                solve_result.enriched_elements.tip_enriched,
                solve_result.enriched_elements.heaviside_enriched,
                mesh,
                solve_result.enriched_elements_triangulation.tip_enriched_triangulation,
                solve_result.enriched_elements_triangulation.heaviside_enriched_triangulation,
                solve_result.u,
                solve_result.node_offset,
                solve_result.enriched_elements.heaviside_enriched_nodes,
                solve_result.enriched_elements.tip_enriched_nodes,
                solve_result.level_set_fields.vertices_level_set_signs,
                LinearTriangle::Triangle13PointRule::gauss_pts,
                LinearTriangle::Triangle13PointRule::gauss_wts,
                solve_result.crack_tip_1_t,
                solve_result.crack_tip_1_n,
                solve_result.crack_tip_2_t,
                solve_result.crack_tip_2_n,
                D,
                E,
                nu,
                Rin,
                Rout
            );

        growth_frames.push_back(
            GrowthFrame{
                crack,
                solve_result,
                k_results
            }
        );

        if (k_results.empty())
        {
            std::cout << "No K results. Stop crack growth.\n";
            break;
        }

        double max_Keq = 0.0;

        for (const TipKResult& r : k_results)
        {
            const double Keq =
                computeEquivalentK(r.K_I, r.K_II);

            max_Keq =
                std::max(max_Keq, Keq);
        }


        const bool grown =
            growCrackOneStep(
                crack,
                k_results,
                KIC,
                da,
                w,
                h
            );

        if (!grown)
        {
            std::cout << "Crack growth finished.\n";
            break;
        }

        saveCrackToFile(
            crack,
            "mesh/crack_iteration_" + std::to_string(growth_step + 1) + ".txt"
        );
    }

    // Важно: добавляем последний кадр уже ПОСЛЕ последнего успешного роста,
    // чтобы финальное положение трещины тоже можно было посмотреть стрелками.
    if (!growth_frames.empty())
    {
        std::cout << "\nResolving final crack frame...\n";

        solve_result =
            solveCrackIteration(
                mesh,
                crack,
                w,
                h,
                wn,
                hn,
                thickness,
                D,
                disable_output,
                disable_debug_output
            );

        std::vector<TipKResult> final_k_results =
            computeStress<13>(
                solve_result.enriched_elements.tip_enriched,
                solve_result.enriched_elements.heaviside_enriched,
                mesh,
                solve_result.enriched_elements_triangulation.tip_enriched_triangulation,
                solve_result.enriched_elements_triangulation.heaviside_enriched_triangulation,
                solve_result.u,
                solve_result.node_offset,
                solve_result.enriched_elements.heaviside_enriched_nodes,
                solve_result.enriched_elements.tip_enriched_nodes,
                solve_result.level_set_fields.vertices_level_set_signs,
                LinearTriangle::Triangle13PointRule::gauss_pts,
                LinearTriangle::Triangle13PointRule::gauss_wts,
                solve_result.crack_tip_1_t,
                solve_result.crack_tip_1_n,
                solve_result.crack_tip_2_t,
                solve_result.crack_tip_2_n,
                D,
                E,
                nu,
                Rin,
                Rout
            );

        growth_frames.push_back(
            GrowthFrame{
                crack,
                solve_result,
                final_k_results
            }
        );
    }

    if (growth_frames.empty())
    {
        throw std::runtime_error("No growth frames were created");
    }

    saveCrackToFile(crack, "mesh/crack_final.txt");

    std::vector<GrowthFrameVisual> growth_visuals;
    growth_visuals.reserve(growth_frames.size());

    for (const GrowthFrame& frame : growth_frames)
    {
        growth_visuals.push_back(
            makeGrowthFrameVisual(
                frame,
                mesh,
                D,
                scale
            )
        );
    }

    const VonMisesScale vm_scale = applyVonMisesColors(growth_visuals);

    const std::vector<Eigen::Vector2d> first_displaced_vertices =
        makeDisplacedVertices(
            mesh,
            growth_frames.front().solve_result,
            scale
        );

    glm::vec2 v1 =
        toGlm(first_displaced_vertices.front().cast<float>().eval());

    glm::vec2 v3 =
        toGlm(first_displaced_vertices.back().cast<float>().eval());

    camera.Position =
        glm::vec3{(v1 + v3) / 2.0f, 1.0f};

    camera.Zoom = 90.0f;
    camera.MovementSpeed = 0.5f;

    GLFWwindow *window;

    if (!glfwInit())
    {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "XFEM crack growth frames", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    checkGLSLVersion();

    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "Current working directory: " << cwd << std::endl;

    std::cout << "\nControls:\n";
    std::cout << "  RIGHT / D : next crack-growth frame\n";
    std::cout << "  LEFT  / A : previous crack-growth frame\n";
    std::cout << "  SPACE     : play / pause\n";
    std::cout << "  R         : restart to frame 0\n";
    std::cout << "  ESC       : close window\n\n";

    Shader xfem_shader("shaders/xfem.vert", "shaders/xfem.frag");
    Shader chain_program("shaders/pchain.vert", "shaders/pchain.frag");
    Shader circle_shader("shaders/circle.vert", "shaders/circle.frag");
    Shader quad_shader("shaders/quad.vert", "shaders/quad.frag");
    Shader overlay_shader("shaders/overlay_2d.vert", "shaders/overlay_2d.frag");
    Shader overlay_text_shader("shaders/overlay_text.vert", "shaders/overlay_text.frag");

    // -----------------------------
    // Crack line VBO / VAO
    // -----------------------------
    GLuint chainVAO = 0;
    GLuint chainVBO = 0;

    glGenVertexArrays(1, &chainVAO);
    glGenBuffers(1, &chainVBO);

    glBindVertexArray(chainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, chainVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void *)offsetof(Vertex, colorPacked));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // -----------------------------
    // Von Mises stress field VBO / VAO
    // Reuses pchain shader layout: vec2 position + packed RGBA color.
    // -----------------------------
    GLuint vonMisesVAO = 0;
    GLuint vonMisesVBO = 0;

    glGenVertexArrays(1, &vonMisesVAO);
    glGenBuffers(1, &vonMisesVBO);

    glBindVertexArray(vonMisesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, vonMisesVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void *)offsetof(Vertex, colorPacked));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // -----------------------------
    // Circle VBO / VAO
    // -----------------------------
    GLuint circleVerticesVBO = 0;
    GLuint circleVBO = 0;
    GLuint circleVAO = 0;
    constexpr int circles_vertices_number = 22;

    {
        constexpr float PI = glm::pi<float>();
        std::vector<glm::vec2> circle_vertices;
        circle_vertices.resize(circles_vertices_number);
        circle_vertices[0].x = 0.0f;
        circle_vertices[0].y = 0.0f;

        for (int i = 1; i < circles_vertices_number; ++i)
        {
            circle_vertices[i].x =
                glm::cos(2.0f * PI / static_cast<float>(circles_vertices_number - 2) * static_cast<float>(i - 1));
            circle_vertices[i].y =
                glm::sin(2.0f * PI / static_cast<float>(circles_vertices_number - 2) * static_cast<float>(i - 1));
        }

        glGenBuffers(1, &circleVerticesVBO);
        glGenBuffers(1, &circleVBO);
        glGenVertexArrays(1, &circleVAO);

        glBindVertexArray(circleVAO);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, circleVerticesVBO);
        glBufferData(
            GL_ARRAY_BUFFER,
            circle_vertices.size() * sizeof(glm::vec2),
            circle_vertices.data(),
            GL_STATIC_DRAW
        );
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Circle), (void *)0);
        glVertexAttribDivisor(1, 1);

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Circle), (void *)sizeof(glm::vec3));
        glVertexAttribDivisor(2, 1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // -----------------------------
    // Quad VBO / VAO
    // -----------------------------
    GLuint quadVerticesVBO = 0;
    GLuint quadVBO = 0;
    GLuint quadVAO = 0;

    {
        std::vector<glm::vec2> rectangle_vertices;
        rectangle_vertices.push_back(glm::vec2(0.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 0.0f));
        rectangle_vertices.push_back(glm::vec2(1.0f, 1.0f));
        rectangle_vertices.push_back(glm::vec2(0.0f, 1.0f));

        glGenBuffers(1, &quadVerticesVBO);
        glGenBuffers(1, &quadVBO);
        glGenVertexArrays(1, &quadVAO);

        glBindVertexArray(quadVAO);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, quadVerticesVBO);
        glBufferData(
            GL_ARRAY_BUFFER,
            rectangle_vertices.size() * sizeof(glm::vec2),
            rectangle_vertices.data(),
            GL_STATIC_DRAW
        );
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)0);
        glVertexAttribDivisor(1, 1);

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v10));
        glVertexAttribDivisor(2, 1);

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v11));
        glVertexAttribDivisor(3, 1);

        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Quad), (void *)offsetof(struct Quad, v01));
        glVertexAttribDivisor(4, 1);

        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Quad), (void *)offsetof(Quad, color));
        glVertexAttribDivisor(5, 1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    std::vector<OverlayVertex> legend_tri_verts;
    std::vector<OverlayVertex> legend_line_verts;
    buildVonMisesLegendGeometry(legend_tri_verts, legend_line_verts);

    LegendGL legend_gl;
    initLegendGL(legend_gl, legend_tri_verts, legend_line_verts);

    // Uses a Windows system sans-serif font. Arial is visually close to the classic ANSYS legend labels.
    FontAtlasGL legend_font = createWindowsFontAtlasGL("Arial", 16);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSwapInterval(1);

    int current_growth_frame = 0;
    bool play_animation = false;
    const double frame_duration = 0.10;
    double last_auto_switch_time = glfwGetTime();
    bool frame_dirty = true;

    std::unordered_set<int> pressed_keys;

    while (!glfwWindowShouldClose(window))
    {
        const float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        bool input_changed_frame = false;

        if (keyPressedOnce(window, GLFW_KEY_RIGHT, pressed_keys) ||
            keyPressedOnce(window, GLFW_KEY_D, pressed_keys))
        {
            if (current_growth_frame + 1 < static_cast<int>(growth_visuals.size()))
            {
                ++current_growth_frame;
                input_changed_frame = true;
            }
        }

        if (keyPressedOnce(window, GLFW_KEY_LEFT, pressed_keys) ||
            keyPressedOnce(window, GLFW_KEY_A, pressed_keys))
        {
            if (current_growth_frame > 0)
            {
                --current_growth_frame;
                input_changed_frame = true;
            }
        }

        if (keyPressedOnce(window, GLFW_KEY_SPACE, pressed_keys))
        {
            play_animation = !play_animation;
            last_auto_switch_time = glfwGetTime();
            std::cout << "Animation: " << (play_animation ? "PLAY" : "PAUSE") << std::endl;
        }

        if (keyPressedOnce(window, GLFW_KEY_R, pressed_keys))
        {
            current_growth_frame = 0;
            play_animation = false;
            input_changed_frame = true;
        }

        if (play_animation)
        {
            const double now = glfwGetTime();

            if (now - last_auto_switch_time >= frame_duration)
            {
                if (current_growth_frame + 1 < static_cast<int>(growth_visuals.size()))
                {
                    ++current_growth_frame;
                }
                else
                {
                    play_animation = false;
                }

                last_auto_switch_time = now;
                input_changed_frame = true;
            }
        }

        if (input_changed_frame)
        {
            std::cout << "Frame " << current_growth_frame + 1
                      << " / " << growth_visuals.size()
                      << std::endl;
            frame_dirty = true;
            draw = true;
        }

        if (frame_dirty)
        {
            uploadGrowthFrameVisual(
                growth_visuals[current_growth_frame],
                quadVBO,
                circleVBO,
                chainVBO,
                vonMisesVBO
            );

            char title[512];
            std::snprintf(
                title,
                sizeof(title),
                "XFEM von Mises | frame %d / %d | min %.3e | max %.3e | Left/Right switch, Space play",
                current_growth_frame + 1,
                static_cast<int>(growth_visuals.size()),
                vm_scale.vmin,
                vm_scale.vmax
            );
            glfwSetWindowTitle(window, title);

            frame_dirty = false;
            draw = true;
        }

        if (!draw)
        {
            glfwWaitEventsTimeout(0.02);
            continue;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        glm::mat4 projection =
            glm::perspective(
                glm::radians(camera.Zoom),
                static_cast<float>(SCR_WIDTH) / static_cast<float>(SCR_HEIGHT),
                0.1f,
                100.0f
            );

        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 MVP = projection * view;

        const GrowthFrameVisual& visual =
            growth_visuals[current_growth_frame];

        quad_shader.use();
        quad_shader.setMat4("mvp", MVP);
        glBindVertexArray(quadVAO);
        glDrawArraysInstanced(
            GL_TRIANGLE_FAN,
            0,
            4,
            static_cast<GLsizei>(visual.quads.size())
        );

        if (!visual.von_mises_vertices.empty())
        {
            chain_program.use();
            chain_program.setMat4("mvp", MVP);
            glBindVertexArray(vonMisesVAO);
            glDrawArrays(
                GL_TRIANGLES,
                0,
                static_cast<GLsizei>(visual.von_mises_vertices.size())
            );
        }

        if (!visual.circles.empty())
        {
            circle_shader.use();
            circle_shader.setMat4("mvp", MVP);
            glBindVertexArray(circleVAO);
            glDrawArraysInstanced(
                GL_TRIANGLE_FAN,
                0,
                circles_vertices_number,
                static_cast<GLsizei>(visual.circles.size())
            );
        }

        if (!visual.crack_vertices.empty())
        {
            chain_program.use();
            chain_program.setMat4("mvp", MVP);
            glBindVertexArray(chainVAO);
            glLineWidth(3.0f);
            glDrawArrays(
                GL_LINE_STRIP,
                0,
                static_cast<GLsizei>(visual.crack_vertices.size())
            );
        }

        glBindVertexArray(0);

        overlay_shader.use();
        drawLegendGL(legend_gl);

        std::vector<TextVertex> legend_text_vertices =
            buildLegendTextVertices(
                legend_font,
                vm_scale,
                current_growth_frame,
                static_cast<int>(growth_visuals.size()),
                static_cast<int>(SCR_WIDTH),
                static_cast<int>(SCR_HEIGHT)
            );

        uploadTextVertices(legend_font, legend_text_vertices);
        overlay_text_shader.use();
        drawTextGL(legend_font);

        glfwSwapBuffers(window);
        draw = false;
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
