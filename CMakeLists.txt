cmake_minimum_required(VERSION 3.18)

project(MicroTensor VERSION 1.0 LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_AVX2 "Enable AVX2 optimizations (x86 only)" ON)
option(ENABLE_NEON "Enable NEON optimizations (ARM only)" ON)
option(ENABLE_TESTS "Build Unit Tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_PROCESSOR)

if(SYSTEM_PROCESSOR MATCHES "amd64|x86_64|i386|i686")
    set(IS_X86 TRUE)
    message(STATUS "Detected Architecture: x86/x64 (${CMAKE_SYSTEM_PROCESSOR})")
elseif(SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(IS_ARM TRUE)
    message(STATUS "Detected Architecture: ARM (${CMAKE_SYSTEM_PROCESSOR})")
else()
    message(STATUS "Detected Architecture: Other (${CMAKE_SYSTEM_PROCESSOR})")
endif()

find_package(OpenMP REQUIRED)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 53 61 62 72 75 86 87 89)
        endif()

        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA detected and enabled.")
    else()
        message(STATUS "No NVCC compiler found. Disabling CUDA support.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# MicroTensor_Lib

set(LIB_SRCS
        include/Tensor.h
        src/Tensor.cpp
        # src/Linear.cpp
        # src/Activation.cpp
        # src/backends/cpu/math_cpu.cpp
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
list(APPEND LIB_SRCS ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)

add_library(MicroTensor_Lib STATIC ${LIB_SRCS})

target_include_directories(MicroTensor_Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(MicroTensor_Lib PUBLIC OpenMP::OpenMP_CXX)

include(CheckCXXCompilerFlag)

if(IS_X86 AND ENABLE_AVX2)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2_FLAG)
        if(HAS_AVX2_FLAG)
            target_compile_options(MicroTensor_Lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
            target_compile_definitions(MicroTensor_Lib PUBLIC USE_AVX2)
            # target_sources(MicroTensor_Lib PRIVATE src/backends/avx/math_avx.cpp)
            message(STATUS "AVX2 optimizations enabled (MSVC).")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" HAS_AVX2_FLAG)
        if(HAS_AVX2_FLAG)
            target_compile_options(MicroTensor_Lib PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
                    $<$<COMPILE_LANGUAGE:CXX>:-mfma>
            )
            target_compile_definitions(MicroTensor_Lib PUBLIC USE_AVX2)
            # target_sources(MicroTensor_Lib PRIVATE src/backends/avx/math_avx.cpp)
            message(STATUS "AVX2 optimizations enabled (GCC/Clang).")
        endif()
    endif()
endif()

if(IS_ARM AND ENABLE_NEON)
    if(NOT MSVC)
        target_compile_options(MicroTensor_Lib PRIVATE -march=native -O3)
        target_compile_definitions(MicroTensor_Lib PUBLIC USE_NEON)
        # target_sources(MicroTensor_Lib PRIVATE src/backends/neon/math_neon.cpp)
        message(STATUS "ARM NEON optimizations enabled.")
    endif()
endif()

if(ENABLE_CUDA)
    target_compile_definitions(MicroTensor_Lib PUBLIC USE_CUDA)
    set_target_properties(MicroTensor_Lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(MicroTensor_Lib PUBLIC CUDA::cudart)

    # target_sources(MicroTensor_Lib PRIVATE
    #     src/backends/cuda/math_cuda.cpp
    #     src/backends/cuda/math_kernels.cu
    # )
endif()

add_executable(MicroTensor app/main.cpp)
target_link_libraries(MicroTensor PRIVATE MicroTensor_Lib)

# Google Test
if(ENABLE_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(UnitTests tests/test_main.cpp tests/test_tensor.cpp)
    target_link_libraries(UnitTests PRIVATE MicroTensor_Lib GTest::gtest_main)

    add_test(NAME AllTests COMMAND UnitTests)
endif()