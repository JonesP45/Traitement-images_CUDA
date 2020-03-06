cmake_minimum_required(VERSION 3.15)
project(TD)

set(CMAKE_CXX_STANDARD 20)

add_executable(TD main.cpp project/blur_seq.cpp project/blur_seq.h)