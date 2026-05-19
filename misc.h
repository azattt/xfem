#pragma once

#include <Eigen/Dense>
#include <glm/glm.hpp>

void checkGLSLVersion();

#ifdef _MSC_VER
#include <intrin.h>
inline bool hasAVX2() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 7) {
        __cpuid(cpuInfo, 7);
        return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
    }
    return false;
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
inline bool hasAVX2() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx))
        return (ebx & (1 << 5)) != 0;
    return false;
}
#else
inline bool hasAVX2() {
    return false;
}
#endif

inline constexpr glm::vec2 toGlm(const Eigen::Vector2f& v) {
    return glm::vec2(v(0), v(1));
}
inline constexpr glm::vec3 toGlm(const Eigen::Vector3f& v) {
    return glm::vec3(v(0), v(1), v(2));
}

// Convert Eigen::Vector4f to glm::vec4
inline constexpr glm::vec4 toGlm(const Eigen::Vector4f& v) {
    return glm::vec4(v(0), v(1), v(2), v(3));
}