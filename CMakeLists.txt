cmake_minimum_required(VERSION 3.16.3)

# Set the compilter to hipcc
set(CMAKE_CXX_COMPILER "hipcc")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find HIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(12_cmake)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

project(queens)

# Include 3rd party libraries
add_subdirectory(
    libs/CLI11
)

find_package(hip REQUIRED)

add_executable(queens
    src/main.cpp
)

target_link_libraries(queens PRIVATE
    CLI11::CLI11
)   
