cmake_minimum_required(VERSION 3.25)
project(makemorepp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(makemore_lib STATIC
        src/makemore.cpp
        include/makemore.h
)

target_include_directories(makemore_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(makemore main.cpp)
target_link_libraries(makemore PRIVATE makemore_lib)
