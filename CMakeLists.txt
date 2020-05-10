cmake_minimum_required(VERSION 3.10.2)
project(ProjetCUDA CUDA)

set(OpenCV_DIR "C:\\opencv\\build")
find_package(OpenCV REQUIRED)
#include_directories("C:\\opencv\\build\\include")

set(CMAKE_CUDA_STANDARD 14)

add_executable(ProjetCUDA
#        Code_bon/sequentiel/main_seq.cu
#        Code_bon/sequentiel_combinaison/main_seq_comb.cu
#        Code_bon/kernel/block32-32/main.cu
#        Code_bon/kernel_combinaison/block32-32/main_comb.cu
        Code_bon/stream/block32-32/main_stream.cu
#        Code_test/shared/main_shared.cu
#        Code_test/stream/stream.cu
        )

set_target_properties(
        ProjetCUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ProjetCUDA ${OpenCV_LIBS})
