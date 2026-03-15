cmake_minimum_required(VERSION 3.25)
project(micrograd++)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(micrograd++ main.cpp)
target_include_directories(micrograd++ PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(micrograd++ PRIVATE -Wall -Wextra -O2)