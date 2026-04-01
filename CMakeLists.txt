cmake_minimum_required(VERSION 3.18)

project(CudaComputerVision
    VERSION 1.0.0
    DESCRIPTION "C++ CUDA Computer Vision Project"
    LANGUAGES CXX CUDA
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# CUDA architectures (adjust for your GPU)
# 75 = Turing (RTX 20xx), 86 = Ampere (RTX 30xx), 89 = Ada Lovelace (RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

# Source files
set(CPP_SOURCES
    src/main.cpp
    src/image_processor.cpp
)

set(CUDA_SOURCES
    kernels/image_filters.cu
    kernels/color_conversion.cu
    kernels/edge_detection.cu
)

set(HEADER_FILES
    include/common.h
    include/cuda_kernels.h
    include/image_processor.h
)

# Create executable
add_executable(${PROJECT_NAME}
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_FILES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link CUDA runtime
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
)

# Compiler warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX-)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# Output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "===========================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================================")
