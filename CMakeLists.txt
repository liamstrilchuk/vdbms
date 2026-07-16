cmake_minimum_required(VERSION 3.15)
project(vdbms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE VDBMS_SOURCES "src/*.cpp")

add_library(vdbms_core STATIC ${VDBMS_SOURCES})
target_include_directories(vdbms_core PUBLIC include)

add_executable(vdbms main.cpp)
target_link_libraries(vdbms PRIVATE vdbms_core)

add_executable(recall_test test/recall.cpp)
target_link_libraries(recall_test PRIVATE vdbms_core)

add_compile_options(-O3)