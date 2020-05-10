#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur2D(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_blur2D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

            sh_blur2D[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
            rgb_out_blur[3 * (row * cols + col) + rgb] = sh_blur2D[3 * (lrow * blockDim.x + lcol) + rgb];
        }
    }
}

__global__ void sharpen2D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_sharpen_2D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (-3 * (h + g + d + b) + 21 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            sh_sharpen_2D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
            rgb_out_sharpen[3 * (row * cols + col) + rgb] = sh_sharpen_2D[3 * (lrow * blockDim.x + lcol) + rgb];
        }
    }
}

__global__ void edge_detect2D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_edge_detect_2D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (9 * (h + g + d + b) - 36 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            sh_edge_detect_2D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = sh_edge_detect_2D[3 * (lrow * blockDim.x + lcol) + rgb];
        }
    }
}


__global__ void blur3D(const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_blur3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

        sh_blur3D[3*(lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        rgb_out_blur[3 * (row * cols + col) + rgb] = sh_blur3D[3 * (lrow * blockDim.x + lcol) + rgb];
    }
}

__global__ void sharpen3D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_sharpen3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (-3 * (h + g + d + b) + 21 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;

        sh_sharpen3D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
        rgb_out_sharpen[3 * (row * cols + col) + rgb] = sh_sharpen3D[3 * (lrow * blockDim.x + lcol) + rgb];
    }
}

__global__ void edge_detect3D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_edge_detect3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;

        sh_edge_detect3D[3*(lrow * blockDim.x + lcol) + rgb] = somme;
        rgb_out_edge_detect[3 * (row * cols + col) + rgb] = sh_edge_detect3D[3*(lrow * blockDim.x + lcol) + rgb];
    }
}


__global__ void blur_edge_detect2D(const unsigned char * rgb_in, unsigned char * rgb_out_edge_detect, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x - 2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y - 2) + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_blur_edge_detect2D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

            sh_blur_edge_detect2D[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }

    __syncthreads();

    auto ww = blockDim.x;

    if (lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1)) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char h = sh_blur_edge_detect2D[3 * ((lrow - 1) * ww + lcol) + rgb];
            unsigned char g = sh_blur_edge_detect2D[3 * (lrow * ww + lcol - 1) + rgb];
            unsigned char c = sh_blur_edge_detect2D[3 * (lrow * ww + lcol) + rgb];
            unsigned char d = sh_blur_edge_detect2D[3 * (lrow * ww + lcol + 1) + rgb];
            unsigned char b = sh_blur_edge_detect2D[3 * ((lrow + 1) * ww + lcol) + rgb];
            int somme = (9 * (h + g + d + b) - 36 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = somme;
        }
    }
}

__global__ void edge_detect_blur2D(const unsigned char * rgb_in, unsigned char * rgb_out_edge_detect, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x - 2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y - 2) + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_edge_detect_blur2D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (int rgb = 0; rgb < 3; ++rgb)
        {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (9 * (h + g + d + b) - 36 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            sh_edge_detect_blur2D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
        }
    }

    __syncthreads();

    auto ww = blockDim.x;

    if (lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1)) {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = sh_edge_detect_blur2D[3 * ((lrow - 1) * ww + lcol - 1) + rgb];
            unsigned char h = sh_edge_detect_blur2D[3 * ((lrow - 1) * ww + lcol) + rgb];
            unsigned char hd = sh_edge_detect_blur2D[3 * ((lrow - 1) * ww + lcol + 1) + rgb];
            unsigned char g = sh_edge_detect_blur2D[3 * (lrow * ww + lcol - 1) + rgb];
            unsigned char c = sh_edge_detect_blur2D[3 * (lrow * ww + lcol) + rgb];
            unsigned char d = sh_edge_detect_blur2D[3 * (lrow * ww + lcol + 1) + rgb];
            unsigned char bg = sh_edge_detect_blur2D[3 * ((lrow + 1) * ww + lcol - 1) + rgb];
            unsigned char b = sh_edge_detect_blur2D[3 * ((lrow + 1) * ww + lcol) + rgb];
            unsigned char bd = sh_edge_detect_blur2D[3 * ((lrow + 1) * ww + lcol + 1) + rgb];

            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }
}


__global__ void blur_edge_detect3D(const unsigned char * rgb_in, unsigned char * rgb_out_blur_edge_detect, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x - 2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y - 2) + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_blur_edge_detect3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

        sh_blur_edge_detect3D[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
    }

    __syncthreads();

    auto ww = blockDim.x;

    if (lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1)) {
        unsigned char h = sh_blur_edge_detect3D[3 * ((lrow - 1) * ww + lcol) + rgb];
        unsigned char g = sh_blur_edge_detect3D[3 * (lrow * ww + lcol - 1) + rgb];
        unsigned char c = sh_blur_edge_detect3D[3 * (lrow * ww + lcol) + rgb];
        unsigned char d = sh_blur_edge_detect3D[3 * (lrow * ww + lcol + 1) + rgb];
        unsigned char b = sh_blur_edge_detect3D[3 * ((lrow + 1) * ww + lcol) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;

        rgb_out_blur_edge_detect[3 * (row * cols + col) + rgb] = somme;
    }
}

__global__ void edge_detect_blur3D(const unsigned char * rgb_in, unsigned char * rgb_out_edge_detect_blur, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x - 2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y - 2) + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_edge_detect_blur3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
//        unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
//        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
//        unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
//        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
//        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
//        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
//        unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
//        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
//        unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];
//
//        sh_edge_detect_blur3D[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;

        sh_edge_detect_blur3D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
    }

    __syncthreads();

    auto ww = blockDim.x;

    if (lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1)) {
//        unsigned char h = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol) + rgb];
//        unsigned char g = sh_edge_detect_blur3D[3 * (lrow * ww + lcol - 1) + rgb];
//        unsigned char c = sh_edge_detect_blur3D[3 * (lrow * ww + lcol) + rgb];
//        unsigned char d = sh_edge_detect_blur3D[3 * (lrow * ww + lcol + 1) + rgb];
//        unsigned char b = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol) + rgb];
//        int somme = (9 * (h + g + d + b) - 36 * c) / 9;
//
//        if (somme > 255) somme = 255;
//        if (somme < 0) somme = 0;
//
//        rgb_out_edge_detect_blur[3 * (row * cols + col) + rgb] = somme;
        unsigned char hg = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol - 1) + rgb];
        unsigned char h = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol) + rgb];
        unsigned char hd = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol + 1) + rgb];
        unsigned char g = sh_edge_detect_blur3D[3 * (lrow * ww + lcol - 1) + rgb];
        unsigned char c = sh_edge_detect_blur3D[3 * (lrow * ww + lcol) + rgb];
        unsigned char d = sh_edge_detect_blur3D[3 * (lrow * ww + lcol + 1) + rgb];
        unsigned char bg = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol - 1) + rgb];
        unsigned char b = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol) + rgb];
        unsigned char bd = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol + 1) + rgb];

        rgb_out_edge_detect_blur[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
    }

    /*auto col = blockIdx.x * (blockDim.x - 2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y - 2) + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh_edge_detect_blur3D[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1) {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;

        sh_edge_detect_blur3D[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
    }

    __syncthreads();

    auto ww = blockDim.x;

    if (lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1)) {
        unsigned char hg = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol - 1) + rgb];
        unsigned char h = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol) + rgb];
        unsigned char hd = sh_edge_detect_blur3D[3 * ((lrow - 1) * ww + lcol + 1) + rgb];
        unsigned char g = sh_edge_detect_blur3D[3 * (lrow * ww + lcol - 1) + rgb];
        unsigned char c = sh_edge_detect_blur3D[3 * (lrow * ww + lcol) + rgb];
        unsigned char d = sh_edge_detect_blur3D[3 * (lrow * ww + lcol + 1) + rgb];
        unsigned char bg = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol - 1) + rgb];
        unsigned char b = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol) + rgb];
        unsigned char bd = sh_edge_detect_blur3D[3 * ((lrow + 1) * ww + lcol + 1) + rgb];

        rgb_out_edge_detect_blur[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
    }*/
}


void main_blur(const dim3 grid, const dim3 block, const unsigned int shared, const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        blur2D <<< grid, block, shared >>>(rgb_in, rgb_out_blur, rows, cols);
    } else {
        blur3D <<< grid, block, shared >>>(rgb_in, rgb_out_blur, rows, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_shared_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_sharpen(const dim3 grid, const dim3 block, const unsigned int shared, const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        sharpen2D <<< grid, block, shared >>> (rgb_in, rgb_out_sharpen, rows, cols);
    } else {
        sharpen3D <<< grid, block, shared >>> (rgb_in, rgb_out_sharpen, rows, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "sharpen_shared_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect(const dim3 grid, const dim3 block, const unsigned int shared, const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        edge_detect2D <<< grid, block, shared >>> (rgb_in, rgb_out_edge_detect, rows, cols);
    } else {
        edge_detect3D <<< grid, block, shared >>> (rgb_in, rgb_out_edge_detect, rows, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_shared_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void main_blur_edge_detect(const dim3 grid, const dim3 block, const unsigned int shared, const unsigned char* rgb_in,
        unsigned char* rgb_out_blur_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        blur_edge_detect2D <<< grid, block, shared >>> (rgb_in, rgb_out_blur_edge_detect, rows, cols);
    } else {
        blur_edge_detect3D <<< grid, block, shared >>> (rgb_in, rgb_out_blur_edge_detect, rows, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "blur_edge_detect_shared_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void main_edge_detect_blur(const dim3 grid, const dim3 block, const unsigned int shared, const unsigned char* rgb_in,
        unsigned char* rgb_out_edge_detect_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        edge_detect_blur2D <<< grid, block, shared >>> (rgb_in, rgb_out_edge_detect_blur, rows, cols);
    } else {
        edge_detect_blur3D <<< grid, block, shared >>> (rgb_in, rgb_out_edge_detect_blur, rows, cols);
    }

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "edge_detect_blur_shared_" << block.x << "-" << block.y << "-" << block.z << ": " << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    // Declarations
    cudaError_t err;

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;
    std::vector<unsigned char> g_blur(taille_rgb);

    std::vector<unsigned char> g_sharpen(taille_rgb);
    std::vector<unsigned char> g_edge_detect(taille_rgb);

    std::vector<unsigned char> g_blur_edge_detect(taille_rgb);
    std::vector<unsigned char> g_edge_detect_blur(taille_rgb);

    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());

    cv::Mat m_out_sharpen(rows, cols, CV_8UC3, g_sharpen.data());
    cv::Mat m_out_edge_detect(rows, cols, CV_8UC3, g_edge_detect.data());

    cv::Mat m_out_blur_edge_detect(rows, cols, CV_8UC3, g_blur_edge_detect.data());
    cv::Mat m_out_edge_detect_blur(rows, cols, CV_8UC3, g_edge_detect_blur.data());

    unsigned char* rgb_in = nullptr;

    unsigned char* rgb_out_blur = nullptr;
    unsigned char* rgb_out_sharpen = nullptr;
    unsigned char* rgb_out_edge_detect = nullptr;

    unsigned char* rgb_out_blur_edge_detect = nullptr;
    unsigned char* rgb_out_edge_detect_blur = nullptr;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMalloc(&rgb_out_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_sharpen, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMalloc(&rgb_out_blur_edge_detect, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMalloc(&rgb_out_edge_detect_blur, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    /////////////////////////////////////////////////////////////////
    ///////////////////// block 32 32 ///////////////////////////////
    /////////////////////////////////////////////////////////////////

    dim3 block_32_32(32, 32); //nb de thread, max 1024
    dim3 grid_32_32(((cols - 1) / (block_32_32.x - 2) + 1), (rows - 1) / (block_32_32.y - 2) + 1);
    unsigned int shared = 3 * block_32_32.x * block_32_32.y;

    // Execution
    main_blur(grid_32_32, block_32_32, shared, rgb_in, rgb_out_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_sharpen(grid_32_32, block_32_32, shared, rgb_in, rgb_out_sharpen, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect(grid_32_32, block_32_32, shared, rgb_in, rgb_out_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    main_blur_edge_detect(grid_32_32, block_32_32, shared, rgb_in, rgb_out_blur_edge_detect, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    main_edge_detect_blur(grid_32_32, block_32_32, shared, rgb_in, rgb_out_edge_detect_blur, rows, cols);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }


    // Recup donnees kernel
    err = cudaMemcpy(g_blur.data(), rgb_out_blur, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(g_sharpen.data(), rgb_out_sharpen, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(g_edge_detect.data(), rgb_out_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMemcpy(g_blur_edge_detect.data(), rgb_out_blur_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
    err = cudaMemcpy(g_edge_detect_blur.data(), rgb_out_edge_detect_blur, taille_rgb, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    cv::imwrite("out_kernel_block_32-32_blur.jpg", m_out_blur);
    cv::imwrite("out_kernel_block_32-32_sharpen.jpg", m_out_sharpen);
    cv::imwrite("out_kernel_block_32-32_edge_detect.jpg", m_out_edge_detect);

    cv::imwrite("out_kernel_block_32-32_blur_edge_detect.jpg", m_out_blur_edge_detect);
    cv::imwrite("out_kernel_block_32-32_edge_detect_blur.jpg", m_out_edge_detect_blur);

    /////////////////////////////////////////////////////////////////
    ///////////////////// block 17 20 3 /////////////////////////////
    /////////////////////////////////////////////////////////////////

//    dim3 block_17_20_3(17, 20, 3); //nb de thread, max 1024
//    dim3 grid_17_20_3(((cols - 1) / (block_17_20_3.x - 2) + 1), (rows - 1) / (block_17_20_3.y - 2) + 1);
//    auto shared = 3 * block_17_20_3.x * block_17_20_3.y;
//
//    // Execution
//    main_blur(grid_17_20_3, block_17_20_3, shared, rgb_in, rgb_out_blur, rows, cols);
//    err = cudaGetLastError();
//    if ( err != cudaSuccess ) { std::cerr << "Error6" << std::endl; }
//    main_sharpen(grid_17_20_3, block_17_20_3, shared, rgb_in, rgb_out_sharpen, rows, cols);
//    err = cudaGetLastError();
//    if ( err != cudaSuccess ) { std::cerr << "Error7" << std::endl; }
//    main_edge_detect(grid_17_20_3, block_17_20_3, shared, rgb_in, rgb_out_edge_detect, rows, cols);
//    err = cudaGetLastError();
//    if ( err != cudaSuccess ) { std::cerr << "Error8" << std::endl; }
//
//    main_blur_edge_detect(grid_17_20_3, block_17_20_3, shared, rgb_in, rgb_out_blur_edge_detect, rows, cols);
//    err = cudaGetLastError();
//    if ( err != cudaSuccess ) { std::cerr << "Error9" << std::endl; }
//    main_edge_detect_blur(grid_17_20_3, block_17_20_3, shared, rgb_in, rgb_out_edge_detect_blur, rows, cols);
//    err = cudaGetLastError();
//    if ( err != cudaSuccess ) { std::cerr << "Error10" << std::endl; }
//
//    // Recup donnees kernel
//    err = cudaMemcpy(g_blur.data(), rgb_out_blur, taille_rgb, cudaMemcpyDeviceToHost);
//    if ( err != cudaSuccess ) { std::cerr << "Error1" << std::endl; }
//    err = cudaMemcpy(g_sharpen.data(), rgb_out_sharpen, taille_rgb, cudaMemcpyDeviceToHost);
//    if ( err != cudaSuccess ) { std::cerr << "Error2" << std::endl; }
//    err = cudaMemcpy(g_edge_detect.data(), rgb_out_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
//    if ( err != cudaSuccess ) { std::cerr << "Error3" << std::endl; }
//
//    err = cudaMemcpy(g_blur_edge_detect.data(), rgb_out_blur_edge_detect, taille_rgb, cudaMemcpyDeviceToHost);
//    if ( err != cudaSuccess ) { std::cerr << "Error4" << std::endl; }
//    err = cudaMemcpy(g_edge_detect_blur.data(), rgb_out_edge_detect_blur, taille_rgb, cudaMemcpyDeviceToHost);
//    if ( err != cudaSuccess ) { std::cerr << "Error5" << std::endl; }
//
//    cv::imwrite("out_kernel_block_17-20-3_blur.jpg", m_out_blur);
//    cv::imwrite("out_kernel_block_17-20-3_sharpen.jpg", m_out_sharpen);
//    cv::imwrite("out_kernel_block_17-20-3_edge_detect.jpg", m_out_edge_detect);
//
//    cv::imwrite("out_kernel_block_17-20-3_blur_edge_detect.jpg", m_out_blur_edge_detect);
//    cv::imwrite("out_kernel_block_17-20-3_edge_detect_blur.jpg", m_out_edge_detect_blur);

    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////

    // Nettoyage memoire
    cudaFree(rgb_in);

    cudaFree(rgb_out_blur);
    cudaFree(rgb_out_sharpen);
    cudaFree(rgb_out_edge_detect);

    cudaFree(rgb_out_blur_edge_detect);
    cudaFree(rgb_out_edge_detect_blur);

    return 0;
}

/*
int main()
{
    // Declarations
    cudaError_t err;

    cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED);
    unsigned char* rgb = m_in.data;
    int rows = m_in.rows;
    int cols = m_in.cols;

    size_t taille_rgb = 3 * rows * cols;
    std::vector<unsigned char> g_blur(taille_rgb);
    cv::Mat m_out_blur(rows, cols, CV_8UC3, g_blur.data());

    unsigned char* rgb_in;

    unsigned char* rgb_out;

    // Init donnes kernel
    err = cudaMalloc(&rgb_in, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMalloc(&rgb_out, taille_rgb);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    err = cudaMemcpy(rgb_in, rgb, taille_rgb, cudaMemcpyHostToDevice);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    dim3 block(32, 32); //nb de thread, max 1024
    dim3 grid(((cols - 1) / (block.x-2) + 1), (rows - 1) / (block.y-2) + 1);

    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    // blur <<< grid, block, block.x*block.y*3 >>> (rgb_in, rgb_out, rows, cols);
    // sharpen <<< grid, block, block.x*block.y*3 >>> (rgb_in, rgb_out, rows, cols);
    // edge_detect <<< grid, block, block.x*block.y*3 >>> (rgb_in, rgb_out, rows, cols);
    blur_and_edge_detect <<< grid, block, block.x*block.y*3 >>> (rgb_in, rgb_out, rows, cols);
    // edge_detect_and_blur <<< grid, block, block.x*block.y*3 >>> (rgb_in, rgb_out, rows, cols);

    // Fin de chrono
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Recup donnees kernel
    err = cudaMemcpy(g_blur.data(), rgb_out, 3 * rows * cols, cudaMemcpyDeviceToHost);
    if ( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

    cv::imwrite("out.jpg", m_out_blur);

    // Nettoyage memoire
    cudaFree(rgb_in);

    cudaFree(rgb_out);

    return 0;
}*/
