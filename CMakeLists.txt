cmake_minimum_required(VERSION 3.18)
project(AiCraft VERSION 1.0 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

# Find CUDA (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    set(CUDA_AVAILABLE ON)
    add_definitions(-DCUDA_AVAILABLE)
    
    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    
    # CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
else()
    message(STATUS "CUDA not found - CPU-only build")
    set(CUDA_AVAILABLE OFF)
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found")
endif()

# Find BLAS/LAPACK (optional for CPU optimization)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)

# Include directories
include_directories(src)
include_directories(src/cuda)

# Source files
set(AICRAFT_SOURCES
    src/tensor.c
    src/training.c
    src/advanced_optimizers.c
    src/graph_optimizer.c
    src/quantization.c
    src/advanced_training.c
    src/advanced_utils.c
    src/benchmark_suite.c
)

# CUDA source files (if available)
if(CUDA_AVAILABLE)
    set(AICRAFT_CUDA_SOURCES
        src/cuda/cuda_kernels.cu
        src/cuda/cuda_utils.cu
        src/cuda/tensor_cuda.cu
        src/cuda/advanced_kernels.cu
    )
endif()

# Create AiCraft library
if(CUDA_AVAILABLE)
    add_library(aicraft STATIC ${AICRAFT_SOURCES} ${AICRAFT_CUDA_SOURCES})
    
    # Link CUDA libraries
    target_link_libraries(aicraft 
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CURAND_LIBRARIES}
    )
    
    # Set CUDA properties
    set_property(TARGET aicraft PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    set_property(TARGET aicraft PROPERTY CUDA_ARCHITECTURES "60;70;75;80;86")
else()
    add_library(aicraft STATIC ${AICRAFT_SOURCES})
endif()

# Link OpenMP if found
if(OpenMP_C_FOUND)
    target_link_libraries(aicraft OpenMP::OpenMP_C)
endif()

# Link math library
target_link_libraries(aicraft m)

# Link BLAS/LAPACK if found
if(BLAS_FOUND)
    target_link_libraries(aicraft ${BLAS_LIBRARIES})
endif()
if(LAPACK_FOUND)
    target_link_libraries(aicraft ${LAPACK_LIBRARIES})
endif()

# Main executable
add_executable(aicraft_demo src/main.c)
target_link_libraries(aicraft_demo aicraft)

# Set executable properties
set_target_properties(aicraft_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS aicraft aicraft_demo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES 
    src/tensor.h
    src/training.h
    src/aicraft_advanced.h
    DESTINATION include/aicraft
)

# Build configuration summary
message(STATUS "")
message(STATUS "AiCraft Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${CUDA_AVAILABLE}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA Version: ${CUDA_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  OpenMP Support: ${OpenMP_C_FOUND}")
message(STATUS "  BLAS Support: ${BLAS_FOUND}")
message(STATUS "  LAPACK Support: ${LAPACK_FOUND}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "")

# Testing (optional)
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    
    # Unit tests
    add_executable(test_tensors tests/test_tensors.c)
    target_link_libraries(test_tensors aicraft)
    add_test(NAME TensorTests COMMAND test_tensors)
    
    add_executable(test_training tests/test_training.c)
    target_link_libraries(test_training aicraft)
    add_test(NAME TrainingTests COMMAND test_training)
endif()

# Benchmarking
option(BUILD_BENCHMARKS "Build benchmark suite" OFF)
if(BUILD_BENCHMARKS)
    add_executable(benchmark_gemm benchmarks/benchmark_gemm.c)
    target_link_libraries(benchmark_gemm aicraft)
    
    add_executable(benchmark_training benchmarks/benchmark_training.c)
    target_link_libraries(benchmark_training aicraft)
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    option(BUILD_DOCS "Build documentation" OFF)
    if(BUILD_DOCS)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in 
                       ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(docs
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
    endif()
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "AiCraft")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Ultra-Optimized Deep Learning Backend")
set(CPACK_PACKAGE_VENDOR "AiCraft Team")
set(CPACK_PACKAGE_CONTACT "support@aicraft.ai")

include(CPack)
