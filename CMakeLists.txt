cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(xfem VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_definitions(-DEIGEN_NO_DEBUG -DEIGEN_MPL2_ONLY -DEIGEN_NO_STATIC_ASSERT)

add_library(xfem_core STATIC
  # compute.cpp
  glad.c
  # eigen.cpp
)
target_include_directories(xfem_core PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(xfem main.cpp fem.cpp gui.cpp levelset.cpp
  Q4.cpp
  postprocess.cpp
  misc.cpp
  BC.cpp
  mp.cpp)

target_link_libraries(xfem xfem_core)

target_include_directories(xfem PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_library(glfw STATIC IMPORTED) # or STATIC instead of SHARED
set_target_properties(glfw PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libs/glfw3.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/GLFW"
)

if(WIN32)
    target_link_libraries(xfem_core glfw opengl32 gdi32)
elseif(UNIX)
    target_link_libraries(xfem_core glfw m pthread dl X11)
endif()



add_library(glm STATIC IMPORTED) # or STATIC instead of SHARED
set_target_properties(glm PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/libs/libglm.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/glm"
)
target_link_libraries(xfem_core glm) 

# # 1. Создаём импортированную цель для Gmsh DLL
# add_library(gmsh SHARED IMPORTED)

# # 2. Указываем расположение файлов
# set_target_properties(gmsh PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
#     IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/libs/gmsh.dll.lib"  # .lib файл
#     IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libs/gmsh.dll"    # .dll файл
# )
# target_link_libraries(xfem_core gmsh)

target_precompile_headers(xfem PRIVATE
  <array>
  <cmath>
  <filesystem>
  <iostream>
  <limits>
  <stdexcept>
  <variant>
  <vector>
  <glad/glad.h>
  <GLFW/glfw3.h>
  <Eigen/Dense>
  <Eigen/Sparse>
  <Eigen/SparseCholesky>
  
  "camera.h"
  "shader.h"
)

# add_custom_command(TARGET xfem POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#         ${CMAKE_CURRENT_SOURCE_DIR}/shaders  # Assuming shaders in 'shaders' dir
#         $<TARGET_FILE_DIR:xfem>/shaders
#     COMMENT "Copying shader files to output directory"
# )



add_custom_target(copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
)
add_custom_target(copy_mesh
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/mesh ${CMAKE_BINARY_DIR}/mesh
)
add_dependencies(xfem copy_assets copy_mesh)


# set(xfem BUILD_TYPE Release)

if(MSVC)
  target_compile_options(xfem PRIVATE /W4 /arch:AVX2) # mb just AVX /O2 /WX  -fsanitize=address
else()
  target_compile_options(xfem PRIVATE -Wall -Wextra -Wpedantic) #   -Werror maybe -mssse3 or -march=native
endif()

# target_compile_options(xfem PRIVATE "-ftime-trace")