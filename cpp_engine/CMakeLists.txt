cmake_minimum_required(VERSION 3.10)
project(studytok_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(include)

add_executable(studytok_engine
    src/main.cpp
    src/csv_parser.cpp
    src/validator.cpp
)
