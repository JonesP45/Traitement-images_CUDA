#include "main_kernel.cuh"

__global__ void blur2D(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (std::size_t rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

            rgb_out_blur[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }
}

__global__ void sharpen2D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (std::size_t rgb = 0; rgb < 3; ++rgb)
        {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (-3 * (h + g + d + b) + 21 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            rgb_out_sharpen[3 * (row * cols + col) + rgb] = somme;
        }
    }
}

__global__ void edge_detect2D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (std::size_t rgb = 0; rgb < 3; ++rgb)
        {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (9 * (h + g + d + b) - 36 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = somme;
        }
    }
}


__global__ void blur3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p1 = mat_in[3 * ((j-1) * cols + i - 1) + k];
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p3 = mat_in[3 * ((j-1) * cols + i + 1) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p7 = mat_in[3 * ((j+1) * cols + i - 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        unsigned char p9 = mat_in[3 * ((j+1) * cols + i + 1) + k];

        mat_out[3 * (j * cols + i) + k] = (p1+p2+p3+p4+p5+p6+p7+p8+p9)/9;
    }
}

__global__ void sharpen3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        int tmp =  (-3*(p2+p4+p6+p8)+21*p5)/9;

        if (tmp > 255) tmp = 255;
        if (tmp < 0) tmp = 0;

        mat_out[3 * (j * cols + i) + k] = tmp;
    }
}

__global__ void edge_detect3D(const unsigned char * mat_in, unsigned char * mat_out, std::size_t cols, std::size_t rows) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto j = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto k = threadIdx.z;

    if (j >= 1 && j < rows - 1 && i >= 1 && i < cols - 1)
    {
        //p1 à p9 correspondent aux 9 pixels à récupérer
        unsigned char p2 = mat_in[3 * ((j-1) * cols + i) + k];
        unsigned char p4 = mat_in[3 * (j * cols + i - 1) + k];
        unsigned char p5 = mat_in[3 * (j * cols + i) + k];
        unsigned char p6 = mat_in[3 * (j * cols + i + 1) + k];
        unsigned char p8 = mat_in[3 * ((j+1) * cols + i) + k];
        int tmp =  (9*(p2+p4+p6+p8)-36*p5)/9;

        if (tmp > 255) tmp = 255;
        if (tmp < 0) tmp = 0;

        mat_out[3 * (j * cols + i) + k] = tmp;
    }
}

int main() {

    return 0;
}