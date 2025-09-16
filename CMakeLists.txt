cmake_minimum_required(VERSION 3.18)
project(optical_digit LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Compiler flags moved to target-specific to avoid leaking MSVC flags into NVCC

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing (75), Ampere (80/86)
add_definitions(-D_FORCE_INLINES)

# Sources
file(GLOB SRCS "*.cpp" "*.cu" "*.hpp")
add_executable(optical_digit ${SRCS})

# Link CUDA and cuFFT
find_package(CUDAToolkit REQUIRED)
target_link_libraries(optical_digit PRIVATE CUDA::cudart CUDA::cufft)

# Target-specific compile options
if (MSVC)
  target_compile_options(optical_digit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
else()
  target_compile_options(optical_digit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math -fno-math-errno -fno-trapping-math>)
  target_compile_options(optical_digit PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
endif()
target_compile_options(optical_digit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
