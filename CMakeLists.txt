cmake_minimum_required(VERSION 3.16)
project(ProjetFini CUDA)

set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ProjetFini
        sequentiel/main_seq.cu
        kernel/main_kernel.cu
        shared/main_shared.cu
        stream/main_stream.cu
        )

set_target_properties(
        ProjetFini
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ProjetFini ${OpenCV_LIBS})