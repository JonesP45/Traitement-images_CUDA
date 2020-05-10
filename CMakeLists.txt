cmake_minimum_required(VERSION 3.10.2)
project(ProjetCUDA CUDA)

set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ProjetCUDA
        Code_final/stream/main_stream.cu

#        Code_final/kernel/main_kernel.cu
#        Code_final/sequentiel/main_seq.cu
#        Code_final/shared/main_shared.cu
        )

set_target_properties(
        ProjetCUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ProjetCUDA ${OpenCV_LIBS})
