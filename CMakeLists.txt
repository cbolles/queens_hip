cmake_minimum_required(VERSION 3.16.3)

# Set HIP path, can be manually set through the HIP_PATH environment variable
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

# Set the compiler to Clang
set(CMAKE_CXX_COMPILER_ID Clang)
find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

project(queens)

set(MY_SOURCE_FILES main.cpp)
set(MY_TARGET_NAME main)
set(MY_HIPCC_OPTIONS)
set(MY_HCC_OPTIONS)
set(MY_NVCC_OPTIONS)

set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})