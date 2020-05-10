#include <opencv2/opencv.hpp>
#include <vector>

__global__ void blur2D(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
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

            sh[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
            rgb_out[3 * (row * cols + col) + rgb] = sh[3*(lrow * blockDim.x + lcol) + rgb];
        }
    }
}

__global__ void sharpen2D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        for (int rgb = 0; rgb < 3; ++rgb)
        {
            unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
            unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
            unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
            unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
            unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
            int somme = (-3 * (h + g + d + b) + 21 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;

            sh[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
            rgb_out_sharpen[3 * (row * cols + col) + rgb] = sh[3 * (lrow * blockDim.x + lcol) + rgb];
        }
    }
}

__global__ void edge_detect2D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh[];

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

            sh[3*(lrow * blockDim.x + lcol) + rgb] = somme;
            rgb_out_edge_detect[3 * (row * cols + col) + rgb] = sh[3*(lrow * blockDim.x + lcol) + rgb];
        }
    }
}


__global__ void blur3D(const unsigned char* rgb_in, unsigned char* rgb_out, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

        sh[3*(lj * blockDim.x + li) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;

        rgb_out[3*(row * cols + col) + rgb] = sh[3*(lj * blockDim.x + li) + rgb];
    }
}

__global__ void sharpen3D(const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (-3 * (h + g + d + b) + 21 * c) / 9;
        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;
        sh[3 * (lj * blockDim.x + li) + rgb] = somme;

        rgb_out_sharpen[3 * (row * cols + col) + rgb] = sh[3 * (lj * blockDim.x + li) + rgb];
    }
}

__global__ void edge_detect3D(const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * blockDim.y + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;
        sh[3*(lj * blockDim.x + li)+rgb] = somme;
        rgb_out_edge_detect[3 * (row * cols + col) + rgb] = sh[3*(lj * blockDim.x + li)+rgb];
    }
}


__global__ void blur_and_edge_detect2D(const unsigned char * rgb_in, unsigned char * rgb_out, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x-2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y-2) + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
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

            sh[3 * (lrow * blockDim.x + lcol) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }

    __syncthreads();

    auto ww = blockDim.x;

    if(lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1) )
    {
        for (int rgb = 0; rgb < 3; ++rgb)
        {
            unsigned char h = sh[3 * ((lrow - 1) * ww + lcol) + rgb];
            unsigned char g = sh[3 * (lrow * ww + lcol - 1) + rgb];
            unsigned char c = sh[3 * (lrow * ww + lcol) + rgb];
            unsigned char d = sh[3 * (lrow * ww + lcol + 1) + rgb];
            unsigned char b = sh[3 * ((lrow + 1) * ww + lcol) + rgb];
            int somme = (9 * (h + g + d + b) - 36 * c) / 9;

            if (somme > 255) somme = 255;
            if (somme < 0) somme = 0;
            rgb_out[3 * (row * cols + col) + rgb] = somme;
        }
    }
}

__global__ void edge_detect_and_blur2D(const unsigned char * rgb_in, unsigned char * rgb_out, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x-2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y-2) + threadIdx.y; //pos de la couleur sur y

    auto lcol = threadIdx.x;
    auto lrow = threadIdx.y;

    extern __shared__ unsigned char sh[];

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

            sh[3 * (lrow * blockDim.x + lcol) + rgb] = somme;
        }
    }

    __syncthreads();

    auto ww = blockDim.x;

    if(lcol > 0 && lcol < (blockDim.x - 1) && lrow > 0 && lrow < (blockDim.y - 1) )
    {
        for (int rgb = 0; rgb < 3; ++rgb) {
            unsigned char hg = sh[3 * ((lrow - 1) * ww + lcol - 1) + rgb];
            unsigned char h = sh[3 * ((lrow - 1) * ww + lcol) + rgb];
            unsigned char hd = sh[3 * ((lrow - 1) * ww + lcol + 1) + rgb];
            unsigned char g = sh[3 * (lrow * ww + lcol - 1) + rgb];
            unsigned char c = sh[3 * (lrow * ww + lcol) + rgb];
            unsigned char d = sh[3 * (lrow * ww + lcol + 1) + rgb];
            unsigned char bg = sh[3 * ((lrow + 1) * ww + lcol - 1) + rgb];
            unsigned char b = sh[3 * ((lrow + 1) * ww + lcol) + rgb];
            unsigned char bd = sh[3 * ((lrow + 1) * ww + lcol + 1) + rgb];

            rgb_out[3 * (row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
        }
    }
}


__global__ void blur_and_edge_detect3D(const unsigned char * rgb_in, unsigned char * rgb_out, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x-2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y-2) + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        unsigned char hg = rgb_in[3 * ((row - 1) * cols + col - 1) + rgb];
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char hd = rgb_in[3 * ((row - 1) * cols + col + 1) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char bg = rgb_in[3 * ((row + 1) * cols + col - 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        unsigned char bd = rgb_in[3 * ((row + 1) * cols + col + 1) + rgb];

        sh[3*(lj * blockDim.x + li) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
    }

    __syncthreads();

    auto ww = blockDim.x;

    if( li > 0 && li < (blockDim.x - 1) && lj > 0 && lj < (blockDim.y - 1) )
    {
        unsigned char h = sh[3 * ((lj - 1) * ww + li) + rgb];
        unsigned char g = sh[3 * (lj * ww + li - 1) + rgb];
        unsigned char c = sh[3 * (lj * ww + li) + rgb];
        unsigned char d = sh[3 * (lj * ww + li + 1) + rgb];
        unsigned char b = sh[3 * ((lj + 1) * ww + li) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;
        rgb_out[3*(row * cols + col) + rgb] = somme;
    }
}

__global__ void edge_detect_and_blur3D(const unsigned char * rgb_in, unsigned char * rgb_out, std::size_t rows, std::size_t cols) {
    auto col = blockIdx.x * (blockDim.x-2) + threadIdx.x; //pos de la couleur sur x
    auto row = blockIdx.y * (blockDim.y-2) + threadIdx.y; //pos de la couleur sur y
    auto rgb = threadIdx.z;

    auto li = threadIdx.x;
    auto lj = threadIdx.y;

    extern __shared__ unsigned char sh[];

    if (row >= 1 && row < rows - 1 && col >= 1 && col < cols - 1)
    {
        unsigned char h = rgb_in[3 * ((row - 1) * cols + col) + rgb];
        unsigned char g = rgb_in[3 * (row * cols + col - 1) + rgb];
        unsigned char c = rgb_in[3 * (row * cols + col) + rgb];
        unsigned char d = rgb_in[3 * (row * cols + col + 1) + rgb];
        unsigned char b = rgb_in[3 * ((row + 1) * cols + col) + rgb];
        int somme = (9 * (h + g + d + b) - 36 * c) / 9;

        if (somme > 255) somme = 255;
        if (somme < 0) somme = 0;
        sh[3*(lj * blockDim.x + li)+rgb] = somme;
    }

    __syncthreads();

    auto ww = blockDim.x;

    if( li > 0 && li < (blockDim.x - 1) && lj > 0 && lj < (blockDim.y - 1) )
    {
        unsigned char hg = sh[3 * ((lj - 1) * ww + li - 1) + rgb];
        unsigned char h = sh[3 * ((lj - 1) * ww + li) + rgb];
        unsigned char hd = sh[3 * ((lj - 1) * ww + li + 1) + rgb];
        unsigned char g = sh[3 * (lj * ww + li - 1) + rgb];
        unsigned char c = sh[3 * (lj * ww + li) + rgb];
        unsigned char d = sh[3 * (lj * ww + li + 1) + rgb];
        unsigned char bg = sh[3 * ((lj + 1) * ww + li - 1) + rgb];
        unsigned char b = sh[3 * ((lj + 1) * ww + li) + rgb];
        unsigned char bd = sh[3 * ((lj + 1) * ww + li + 1) + rgb];

        rgb_out[3*(row * cols + col) + rgb] = (hg + h + hd + g + c + d + bg + b + bd) / 9;
    }
}


void main_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        blur2D <<< grid, block >>>(rgb_in, rgb_out_blur, rows, cols);
    } else {
        blur3D <<< grid, block >>>(rgb_in, rgb_out_blur, rows, cols);
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

void main_sharpen(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_sharpen, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        sharpen2D <<< grid, block >>> (rgb_in, rgb_out_sharpen, rows, cols);
    } else {
        sharpen3D <<< grid, block >>> (rgb_in, rgb_out_sharpen, rows, cols);
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

void main_edge_detect(const dim3 grid, const dim3 block, const unsigned char* rgb_in, unsigned char* rgb_out_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        edge_detect2D <<< grid, block >>> (rgb_in, rgb_out_edge_detect, rows, cols);
    } else {
        edge_detect3D <<< grid, block >>> (rgb_in, rgb_out_edge_detect, rows, cols);
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


void main_blur_edge_detect(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
                           unsigned char* rgb_tmp_blur_edge_detect, unsigned char* rgb_out_blur_edge_detect, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        blur2D <<< grid, block >>> (rgb_in, rgb_out_blur_edge_detect, rows, cols);
        edge_detect2D <<< grid, block >>> (rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);
    } else {
        blur3D <<< grid, block >>> (rgb_in, rgb_tmp_blur_edge_detect, rows, cols);
        edge_detect3D <<< grid, block >>> (rgb_tmp_blur_edge_detect, rgb_out_blur_edge_detect, rows, cols);
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

void main_edge_detect_blur(const dim3 grid, const dim3 block, const unsigned char* rgb_in,
                           unsigned char* rgb_tmp_edge_detect_blur, unsigned char* rgb_out_edge_detect_blur, int rows, int cols) {
    // Debut de chrono
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Appel kernel
    if (block.z == 1) {
        edge_detect2D <<< grid, block >>> (rgb_in, rgb_tmp_edge_detect_blur, rows, cols);
        blur2D <<< grid, block >>> (rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);
    } else {
        edge_detect3D <<< grid, block >>> (rgb_in, rgb_tmp_edge_detect_blur, rows, cols);
        blur3D <<< grid, block >>> (rgb_tmp_edge_detect_blur, rgb_out_edge_detect_blur, rows, cols);
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
}