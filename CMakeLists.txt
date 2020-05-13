cmake_minimum_required(VERSION 3.10.2)
project(ProjetCUDA CUDA)

set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ProjetCUDA
#        Code_final/sequentiel/main_seq.cu
#        Code_final/kernel/main_kernel.cu
#        Code_final/shared/main_shared.cu
#        Code_final/stream/main_stream.cu
#        test/sequentiel/main_seq.cu
#        test/kernel/main_kernel.cu
#        test/shared/main_shared.cu
#        test/stream/main_stream.cu
        test/shared_stream/main_shared_stream.cu
        )

set_target_properties(
        ProjetCUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ProjetCUDA ${OpenCV_LIBS})
