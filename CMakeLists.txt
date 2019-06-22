cmake_minimum_required(VERSION 3.2)
project(daecpp)

# Find the BLAS stuff
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(BLA_DEFINITIONS)

if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()

find_package(MKL REQUIRED)

if (MKL_FOUND)
    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLA_DEFINITIONS USE_MKL)
endif()

message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

# dae-cpp
set(dae_root third-party/dae-cpp)
FILE(GLOB DAE_CPP_SOURCES ${dae_root}/src/*.cpp)
FILE(GLOB DAE_CPP_INCLUDES ${dae_root}/src/*.h)

add_library(daecpp SHARED ${DAE_CPP_SOURCES})

target_compile_definitions(daecpp PUBLIC ${BLA_DEFINITIONS})
target_link_libraries(daecpp PUBLIC ${BLAS_LIBRARIES})
target_include_directories(daecpp PUBLIC
  ${BLA_INCLUDE_DIR}
  ${dae_root}/src
  )

add_subdirectory(third-party/pybind11)
pybind11_add_module(pydae src/dae-wrapper.cpp)

target_link_libraries(pydae PRIVATE daecpp)
