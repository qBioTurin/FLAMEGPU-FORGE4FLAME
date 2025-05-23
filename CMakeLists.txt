# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Optionaly set the version of flamegpu which should be used, ideally a tag (i.e. `v2.0.0-rc`) or branch name, or potentially a commit hash.
set(FLAMEGPU_VERSION "5cbf796d4b9e8fea7cff3bc178e05115a3d7a07b" CACHE STRING "FLAMEGPU/FLAMEGPU2 git branch or tag to use")
# If the above version is a hash instead, also set FLAMEGPU_VERSION_ALLOW_HASH to ON
# set(FLAMEGPU_VERSION_ALLOW_HASH "ON")

# Run to obtain a checkpoint minimizing the cost of the simulation
option(CHECKPOINT "Run to obtain a checkpoint" OFF)
option(DEBUG "Run with debug info" OFF)
if(CHECKPOINT)
    add_definitions(-DCHECKPOINT)
endif()
if(DEBUG)
    add_definitions(-DDEBUG)
endif()

# Manually specify the FLAMEGPU_VISUALISATION option to provide it prior to original configuration and allow the default to be overridden in the downstream project
option(FLAMEGPU_VISUALISATION "Enable FLAMEGPU visualisation support" OFF)
if(FLAMEGPU_VISUALISATION)
    add_definitions(-DFLAMEGPU_VISUALISATION)
endif()

add_definitions(-D GLM_ENABLE_EXPERIMENTAL)

# Our core dependency is FLAMEGPU2 lib, first lets find it
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flamegpu2.cmake)

# Handle CMAKE_CUDA_ARCHITECTURES gracefully, passign the project name for code-injection
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)
flamegpu_init_cuda_architectures(PROJECT FLAMEGPUABM)

# Name the project and set languages, this must be done after flamegpu_init_cuda_architectures
project(FLAMEGPUABM CUDA CXX)

# Include common rules from the FLAMEGPU/FLAMEGPU2 repositories CMake
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of binary files
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)

# Manually list all of the files which require building or influence re-building (i.e. headers).
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/autogenerated_defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/variables_names.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/autogenerated_variables_names.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/agent_functions.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host_functions.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device_functions.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pugixml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pugixml.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu
)

# Add the executable and set required flags for the target
flamegpu_add_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)

# Add src directory to include path
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Also set as (visual studio) startup project
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")