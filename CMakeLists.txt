cmake_minimum_required(VERSION 3.15)
project(vdbms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set(OPT_FLAGS -O3 -march=native -ffast-math -fno-finite-math-only -mavx2 -mfma)

file(GLOB_RECURSE VDBMS_SOURCES "src/*.cpp")

add_library(vdbms_core STATIC ${VDBMS_SOURCES})
target_include_directories(vdbms_core PUBLIC include)
target_compile_options(vdbms_core PRIVATE ${OPT_FLAGS})

add_executable(vdbms main.cpp)
target_link_libraries(vdbms PRIVATE vdbms_core)
target_compile_options(vdbms PRIVATE ${OPT_FLAGS})

add_executable(recall_test test/recall.cpp)
target_link_libraries(recall_test PRIVATE vdbms_core)
target_compile_options(recall_test PRIVATE ${OPT_FLAGS})