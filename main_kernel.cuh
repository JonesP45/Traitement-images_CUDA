#ifndef PROJETCUDA_MAIN_KERNEL_CUH
#define PROJETCUDA_MAIN_KERNEL_CUH

void blur(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols);

void sharpen(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols);

void edge_detect(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols);


__global__ void blur2D(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols);

__global__ void sharpen2D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols);

__global__ void edge_detect2D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols);


__global__ void blur3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows);

__global__ void sharpen3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows);

__global__ void edge_detect3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows);


void main_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_blur,
               int rows, int cols);

void main_sharpen(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_sharpen,
                  int rows, int cols);

void main_edge_detect(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect,
                      int rows, int cols);


void main_blur_edge_detect(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
                           unsigned char* rgb_tmp_blur_edge_detect, unsigned char* rgb_out_blur_edge_detect,
                           int rows, int cols);

void main_edge_detect_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
                           unsigned char* rgb_tmp_edge_detect_blur, unsigned char* rgb_out_edge_detect_blur,
                           int rows, int cols);


int main_kernel();

#endif //PROJETCUDA_MAIN_KERNEL_CUH
